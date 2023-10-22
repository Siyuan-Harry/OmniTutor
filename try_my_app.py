import pandas as pd
import numpy as np
import faiss
from transformers import pipeline
import openai
import tempfile
from sentence_transformers import SentenceTransformer
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def chunkstring(string, length):
        return (string[0+i:length+i] for i in range(0, len(string), length))

def summarize_materials(file_paths): #这里的重点是，对每一个file做尽可能简短且覆盖全面的summarization
    # initialize text summarization pipeline
    summarizer = pipeline("summarization") #调用transformers来对Knowledge Materials（md文档）做summarization，缺陷是效率比较低可能会搞很久
    summarized_materials = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = file.read()
            chunks = list(chunkstring(data, 1024)) #chunkstring是自定义的函数
            summary = [summarizer(chunk, max_length=64, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
            summarized_materials.append((f"Summary for {file_path}:\n {' '.join(summary)}\n"))

    return summarized_materials

def get_completion_from_messages(messages, model="gpt-4", temperature=0):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]

#调用gpt API生成课程大纲 + 每节课解释，随后输出为md文档。并在课程内一直保留着
def genarating_outline(summarized_materials, num_lessons):

    system_message = 'You are a great AI teacher and linguist, skilled at create course outline based on summarized knowledge materials.'
    user_message = f"""You are a great AI teacher and linguist,
            skilled at generating course outline based on summarized knowledge materials.
            Based on knowledge materials provided, you should carefully design a course by outputting its outline.
            This course is aimed to teach new hands the related knowledge focus on these materials.
            knowledge materials: {summarized_materials}
            you should output course outline in a python list format, Do not include anything else except that python list in your output.
            Example output format:
            [[name_lesson1, abstract_lesson1],[name_lesson2, abstrct_lesson2]]
            for this course, you should output {num_lessons} lessons in total.
            """
    messages =  [
                {'role':'system',
                'content': system_message},
                {'role':'user',
                'content': user_message},
            ]

    response = get_completion_from_messages(messages)

    list_response = ['nothing in the answers..']

    try:
        list_response = eval(response)
    except SyntaxError:
        pass

    return list_response

def courseOutlineGenerating(file_paths, num_lessons):
    summarized_materials = summarize_materials(file_paths)
    course_outline = genarating_outline(summarized_materials, num_lessons)
    return course_outline

def constructVDB(file_paths):
#把KM拆解为chunks

    chunks = []
    for filename in file_paths:
        with open(filename, 'r') as f:
            content = f.read()
            for chunk in chunkstring(content, 500):
                chunks.append(chunk)
    chunk_df = pd.DataFrame(chunks, columns=['chunk'])

    #从文本chunks到embeddings
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    embeddings = model.encode(chunk_df['chunk'].tolist())
    # convert embeddings to a dataframe
    embedding_df = pd.DataFrame(embeddings.tolist())
    # Concatenate the original dataframe with the embeddings
    paraphrase_embeddings_df = pd.concat([chunk_df, embedding_df], axis=1)
    # Save the results to a new csv file

    #从embeddings到向量数据库
    # Load the embeddings
    data = paraphrase_embeddings_df
    embeddings = data.iloc[:, 1:].values  # All columns except the first (chunk text)

    # Ensure that the array is C-contiguous
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    # Preparation for Faiss
    dimension = embeddings.shape[1]  # the dimension of the vector space
    index = faiss.IndexFlatL2(dimension)
    # Normalize the vectors
    faiss.normalize_L2(embeddings)
    # Build the index
    index.add(embeddings)
    # write index to disk
    return paraphrase_embeddings_df, index

def searchVDB(search_sentence, paraphrase_embeddings_df, index):
    #从向量数据库中检索相应文段
    data = paraphrase_embeddings_df
    embeddings = data.iloc[:, 1:].values  # All columns except the first (chunk text)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    sentence_embedding = model.encode(search_sentence)

    # Ensuring the sentence embedding is in the correct format
    sentence_embedding = np.ascontiguousarray([sentence_embedding], dtype=np.float32)
    # Searching for the top 3 nearest neighbors in the FAISS index
    D, I = index.search(sentence_embedding, k=3)
    # Printing the top 3 most similar text chunks
    retrieved_chunks_list = []

    for idx in I[0]:
        retrieved_chunks_list.append(data.iloc[idx].chunk)

    return retrieved_chunks_list

def generateCourse(topic, materials):

    #调用gpt4 API生成一节课的内容
    system_message = 'You are a great AI teacher and linguist, skilled at generating course content based on given lesson topic and knowledge materials.'

    user_message = f"""You are a great AI teacher and linguist,
            skilled at generating course content based on given lesson topic and knowledge materials.
            Your lesson topic is within the 「」 quotes, and the knowledge materials are within the 【】 brackets.
            lesson topic: 「{topic}」,
            knowledge materials：【{materials} 】"""

    messages =  [
                {'role':'system',
                'content': system_message},
                {'role':'user',
                'content': user_message},
            ]

    response = get_completion_from_messages(messages)
    return response


def main(file_paths, num_lessons):
    #该程序要实现的目的：根据用户上传的文档（小于等于3个，每个文档不超过xx字），自动地为他生成，可以教会他的课程内容，保存为md文档。

    courseOutline = courseOutlineGenerating(file_paths, num_lessons)
    embeddings_df, faiss_index = constructVDB(file_paths)
    num = 0
    course_content_list = []
    for lesson in courseOutline:
        num += 1
        retrievedChunksList = searchVDB(lesson, embeddings_df, faiss_index)
        courseContent = generateCourse(lesson, retrievedChunksList)
        course_content_list.append(courseContent)

    return courseOutline, course_content_list

def app():
    st.title("OmniTutor v0.0.1")

    with st.sidebar:
        st.image("https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231021212525.png")
        added_files = st.file_uploader('Upload .txt or .md file', type=['md'], accept_multiple_files=True)
        num_lessons = st.slider('How many lessons do you want this course to have?', min_value=2, max_value=9, value=3, step=1)
        btn = st.button('submit')
    
    Course_Outline = st.text_area("Course Outline")
    Course_Content = st.text_area("Course Content")

    if btn:
        temp_file_paths = []
        with st.spinner("Processing file..."):
            for added_file in added_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
                    tmp.write(added_file.getvalue())
                    tmp_path = tmp.name
                    temp_file_paths.append(tmp_path)
        courseOutline, course_content_list = main(temp_file_paths, num_lessons)
        Course_Outline.value = courseOutline
        Course_Content.value = course_content_list
        
    
    #col1, col2 = st.columns(2)

    
if __name__ == "__main__":
    app()