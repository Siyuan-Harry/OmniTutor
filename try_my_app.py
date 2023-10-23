import pandas as pd
import numpy as np
import faiss
import openai
import tempfile
from sentence_transformers import SentenceTransformer
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def download_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

def chunkstring(string, length):
        return (string[0+i:length+i] for i in range(0, len(string), length))

def get_keywords(file_paths): #这里的重点是，对每一个file做尽可能简短且覆盖全面的summarization
    download_nltk()
    keywords_list = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = file.read()
            # tokenize
            words = word_tokenize(data)
            # remove punctuation
            words = [word for word in words if word.isalnum()]
            # remove stopwords
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]
            # lemmatization
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
            # count word frequencies
            word_freq = Counter(words)
            # get top 20 most common words
            keywords = word_freq.most_common(20)
            new_keywords = []
            for word in keywords:
                new_keywords.append(word[0])
            str_keywords = ''
            for word in new_keywords:
                str_keywords += word + ", "
            keywords_list.append(f"Top20 frequency keywords for {file_path}: {str_keywords}")

    return keywords_list

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
            skilled at generating course outline based on keywords of the course.
            Based on keywords provided, you should carefully design a course by outputting its outline.
            Through learning this course, learner should understand those key concepts.
            keywords: {summarized_materials}
            you should output course outline in a python list format, Do not include anything else except that python list in your output.
            Example output format:
            [[name_lesson1, abstract_lesson1],[name_lesson2, abstrct_lesson2]]
            for this course, you should design {num_lessons} lessons in total.
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
    summarized_materials = get_keywords(file_paths)
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
    sentence_embedding = model.encode([search_sentence])

    # Ensuring the sentence embedding is in the correct format
    sentence_embedding = np.ascontiguousarray(sentence_embedding, dtype=np.float32)
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
            Your lesson topic and abstract is within the 「」 quotes, and the knowledge materials are within the 【】 brackets.
            lesson topic and abstract: 「{topic}」,
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
        added_files = st.file_uploader('Upload .md file', type=['.md'], accept_multiple_files=True)
        num_lessons = st.slider('How many lessons do you want this course to have?', min_value=2, max_value=9, value=3, step=1)
        btn = st.button('submit')
    
    
    col1, col2 = st.columns(2)

    with col1:
        st.text_area("Course Outline")
        st.text_area("Course Content")
        if btn:
            temp_file_paths = []
            with st.spinner("Processing file..."):
                for added_file in added_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
                        tmp.write(added_file.getvalue())
                        tmp_path = tmp.name
                        temp_file_paths.append(tmp_path)
            courseOutline, course_content_list = main(temp_file_paths, num_lessons)

            st.text_area("Course Outline", value=courseOutline)
            st.text_area("Course Content", value=course_content_list)

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    #这里的session.state就是保存了这个对话会话的一些基本信息和设置

    prompt = st.chat_input("What is up?")
        # Add user message to chat history

    with col2:
        st.title("chatbot assistant")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    
if __name__ == "__main__":
    app()