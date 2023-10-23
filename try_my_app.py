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
import time

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
def genarating_outline(keywords, num_lessons):

    system_message = 'You are a great AI teacher and linguist, skilled at create course outline based on summarized knowledge materials.'
    user_message = f"""You are a great AI teacher and linguist,
            skilled at generating course outline based on keywords of the course.
            Based on keywords provided, you should carefully design a course outline. 
            Requirements: Through learning this course, learner should understand those key concepts.
            Key concepts: {keywords}
            you should output course outline in a python list format, Do not include anything else except that python list in your output.
            Example output format:
            [[name_lesson1, abstract_lesson1],[name_lesson2, abstrct_lesson2]]
            In the example, you can see each element in this list consists of two parts: the "name_lesson" part is the name of the lesson, and the "abstract_lesson" part is the one-sentence description of the lesson, intruduces knowledge it contained. 
            for each lesson in this course, you should provide these two information and organize them as exemplified.
            for this course, you should design {num_lessons} lessons in total.
            Start the work now.
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
            for chunk in chunkstring(content, 1024):
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
    system_message = 'You are a great AI teacher and linguist, skilled at writing informative and easy-to-understand course script based on given lesson topic and knowledge materials.'

    user_message = f"""You are a great AI teacher and linguist,
            skilled at writing informative and easy-to-understand course script based on given lesson topic and knowledge materials.
            You should write a course for new hands, they need detailed and vivid explaination to understand the topic. 
            Here are general steps of creating a well-designed course. Please follow them step-by-step:
            Step 1. Write down the teaching purpose of the lesson initially in the script.
            Step 2. Write down the outline of this lesson (outline is aligned to the teaching purpose), then follow the outline to write the content.
            Step 3. Review the content,add some examples (including code example) to the core concepts of this lesson, making sure examples are familiar with learner. Each core concepts should at least with one example.
            Step 4. Review the content again, make some analogies or metaphors to the concepts that come up frequently to make the explanation of them more easier to understand.
            Make sure all these steps are considered when writing the lesson script content.
            Your lesson topic and abstract is within the 「」 quotes, and the knowledge materials are within the 【】 brackets.
            lesson topic and abstract: 「{topic}」,
            knowledge materials related to this lesson：【{materials} 】
            Start writting the script of this lesson now.
            """

    messages =  [
                {'role':'system',
                'content': system_message},
                {'role':'user',
                'content': user_message},
            ]

    response = get_completion_from_messages(messages)
    return response

def app():
    st.title("OmniTutor v0.0.1")

    with st.sidebar:
        st.image("https://siyuan-harry.oss-cn-beijing.aliyuncs.com/oss://siyuan-harry/20231021212525.png")
        added_files = st.file_uploader('Upload .md file', type=['.md'], accept_multiple_files=True)
        num_lessons = st.slider('How many lessons do you want this course to have?', min_value=5, max_value=20, value=10, step=1)
        btn_outline = st.button('submit')
    
    
    col1, col2 = st.columns([0.6,0.4], gap='large')

    with col1:
        
        if btn_outline:
            temp_file_paths = []
            file_proc_state = st.text("Processing file...")
            for added_file in added_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
                    tmp.write(added_file.getvalue())
                    tmp_path = tmp.name
                    temp_file_paths.append(tmp_path)
            file_proc_state.text("Processing file...Done")

            outline_generating_state = st.text("Generating Course Oueline...")
            course_outline_list = courseOutlineGenerating(temp_file_paths, num_lessons)
            outline_generating_state.text("Generating Course Oueline...Done")

            course_outline_string = ''
            outline_area = st.text_area("Course Outline", value=course_outline_string) #检查下可以写到这里不，如果是空值就写到循环后面
            lessons_count = 0
            for outline in course_outline_list:
                lessons_count += 1
                course_outline_string += f"{lessons_count}." + outline[0] + '\n'
                course_outline_string += outline[1] + '\n\n'
                time.sleep(1)
                outline_area.value = course_outline_string

            vdb_state = st.text("Constructing vector database from provided materials...")
            embeddings_df, faiss_index = constructVDB(file_paths)
            vdb_state.text("Constructing vector database from provided materials...Done")
            
            count_generating_content = 0
            for lesson in course_outline_list:
                count_generating_content += 1
                content_generating_state = st.text(f"Writing content for lesson {count_generating_content}...")
                retrievedChunksList = searchVDB(lesson, embeddings_df, faiss_index)
                courseContent = generateCourse(lesson, retrievedChunksList)
                content_generating_state.text(f"Writing content for lesson {count_generating_content}...Done")
                st.text_area("Course Content", value=courseContent)

    prompt = st.chat_input("Enter your questions when learning...")
        # Add user message to chat history

    with col2:
        st.caption(''':blue[AI Assistant]: Ask this TA any questions related to this course and get direct answers. :sunglasses:''')
            # Set a default model

        with st.chat_message("assistant"):
            st.write("Hello👋, how can I help you today? 😄")
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