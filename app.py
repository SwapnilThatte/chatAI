import streamlit as st
from run_model import generate_response, generate_RAG_response
from web_search import search_web
from deep_research import perform_deep_research
import tempfile

st.set_page_config(layout="wide")

def main():
    st.title("💬 Chat with Gemma")
    

    with st.sidebar:
        st.title("Tools")
        st.markdown("For source code of this website, visit : <some_link>")
        
        option = st.selectbox(
            "Choose tools",
            ("Simple Chat", "Web Search", "Upload PDF", "Deep Web Search"),
        )

        temperature = st.slider(
            label="Temperature (controls randomness)",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.01,
            help="Lower = more deterministic, Higher = more random"
        )
        # Top-k sampling
        top_k = st.slider(
            label="Top-k (limits to top K tokens by probability)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="0 = disable top-k filtering"
        )

        # Top-p (nucleus sampling)
        top_p = st.slider(
            label="Top-p (nucleus sampling cutoff)",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.01,
            help="0.0 = conservative, 1.0 = more random"
        )

        if option == "Upload PDF":
            file = st.file_uploader(label="Uploaded file will provide context to LLM", type="pdf")
            

        if option == "Web Search":
            st.write("Web Search Enabled for next query")
            # WB_SEARCH = True

        if option == "None":
            st.warning("You are not using any tool")

        if option == "Deep Web Search":
            st.write("Deep Web Research Enabled for next query")



    # col1, col2 = st.columns([6, 1], gap="small")

    # column 1
    # with col1:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # prompt = st.chat_input("Say something...")

    if prompt:= st.chat_input("Say something..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        if option == "Simple Chat":

            response = generate_response(history=st.session_state.messages, query=prompt, temperature=temperature, top_k=top_k, top_p=top_p)
            st.chat_message("assistant").markdown(response)
            # st.session_state.messages.append({"role": "assistant", "content": response})
        
        if option == "Web Search":
            st.session_state.messages.append([{"role": "user", "content" : prompt}])

            response, sources = search_web(prompt)
            # asnswer = response
            # st.chat_message("assistant").markdown(f"{response}\n\n###Sources\n{'\n'.join([source for source in sources])}")
            with st.chat_message("assistant"):
                st.markdown(f"{response}\n\n### Sources\n" + "\n".join(sources))

        if option == "Deep Web Search":
            st.session_state.messages.append([{"role": "user", "content" : prompt}])

            response = perform_deep_research(prompt)
            # asnswer = response
            # st.chat_message("assistant").markdown(f"{response}\n\n###Sources\n{'\n'.join([source for source in sources])}")
            with st.chat_message("assistant"):
                st.markdown(response)

        if option == "Upload PDF":
            st.session_state.messages.append([{"role": "user", "content" : prompt}])

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            print(tmp_path)
            response = generate_RAG_response(prompt, tmp_path, st.session_state.messages)
            # print(file)
            with st.chat_message("assistant"):
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        
    # # Slider in column 2
    # with col2:
    #     temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    #     top_k = st.slider("Top k", 0.0, 100.0, 40.0)
    #     top_p = st.slider("Top p", 0.0, 1.0, 0.95)


if __name__ == "__main__":
    main()