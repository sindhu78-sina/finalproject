"""AI Peer Learning Matcher - Streamlit + sentence-transformers + ChromaDB + Ollama.
Single file. Uses only: streamlit, pandas, requests, sentence_transformers, chromadb."""
import os
import uuid

import chromadb
import pandas as pd
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://localhost:11434"
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "students.csv")
COLS = ["name", "branch", "strong_in", "need_help_in", "availability", "style"]

SAMPLE = [
    ["Aarav", "CSE", "Python, Data Structures, Recursion", "Engineering Chemistry, Calculus", "Evenings", "Problem solving"],
    ["Diya", "CSE", "Mathematics, Calculus, Linear Algebra", "Python, Data Structures", "Weekends", "Step-by-step"],
    ["Rohan", "ECE", "Engineering Chemistry, Physics", "English Communication, Python", "Mornings", "Discussion"],
    ["Meera", "IT", "English Communication, Presentation Skills", "Mathematics, Data Structures", "Evenings", "Practice tests"],
    ["Kabir", "CSE", "Data Structures, Algorithms, C Programming", "Engineering Chemistry, Physics", "Weekends", "Problem solving"],
    ["Ananya", "ECE", "Mathematics, Probability, Physics", "Python, English Communication", "Evenings", "Step-by-step"],
    ["Vihaan", "ME", "Engineering Chemistry, Engineering Drawing", "Mathematics, Python", "Mornings", "Visual diagrams"],
    ["Isha", "IT", "Python, Web Development, SQL", "Mathematics, Physics", "Weekends", "Project-based"],
]


# ---------------- data ----------------
def load_students():
    if not os.path.exists(CSV):
        pd.DataFrame(SAMPLE, columns=COLS).to_csv(CSV, index=False)
    return pd.read_csv(CSV)


# ---------------- AI matching (sentence-transformers + ChromaDB) ----------------
@st.cache_resource(show_spinner="Loading AI model (first time only)...")
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts):
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def find_matches(me, df, top_n=5):
    peers = df[df["name"].str.lower() != me["name"].lower()].reset_index(drop=True)
    if peers.empty:
        return peers
    n = len(peers)
    ids = [str(i) for i in range(n)]
    client = chromadb.EphemeralClient()
    tag = uuid.uuid4().hex[:8]
    strengths = client.create_collection(f"s{tag}")
    needs = client.create_collection(f"n{tag}")
    strengths.add(ids=ids, embeddings=embed(peers["strong_in"].astype(str).tolist()))
    needs.add(ids=ids, embeddings=embed(peers["need_help_in"].astype(str).tolist()))

    def sims(col, text):
        res = col.query(query_embeddings=embed([text]), n_results=n)
        # vectors are normalised, so cosine similarity = 1 - squared_L2 / 2
        return {int(i): 1 - d / 2 for i, d in zip(res["ids"][0], res["distances"][0])}

    a = sims(strengths, me["need_help_in"])   # they are strong where I am weak
    b = sims(needs, me["strong_in"])          # I am strong where they are weak
    peers["they_help_me"] = [round(max(a[i], 0) * 100, 1) for i in range(n)]
    peers["i_help_them"] = [round(max(b[i], 0) * 100, 1) for i in range(n)]
    peers["match_score"] = (0.6 * peers["they_help_me"] + 0.4 * peers["i_help_them"]).round(1)

    client.delete_collection(f"s{tag}")
    client.delete_collection(f"n{tag}")
    return peers.sort_values("match_score", ascending=False).head(top_n)


# ---------------- Ollama chatbot ----------------
def ollama_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None  # None = Ollama not running


SYSTEM = ("You are a friendly peer-learning assistant for first-year engineering students. "
          "Help plan study sessions with their matched peer, explain concepts simply, and "
          "suggest icebreakers and study schedules. Keep answers short and clear.")


def ask(model, history, context):
    msgs = [{"role": "system", "content": SYSTEM + "\n" + context}] + history
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat",
                          json={"model": model, "messages": msgs, "stream": False}, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"Could not get a reply from Ollama: {e}"


# ---------------- UI ----------------
st.set_page_config(page_title="AI Peer Learning Matcher", page_icon="🤝", layout="wide")
st.session_state.setdefault("students", load_students())
st.session_state.setdefault("chat", [])
st.session_state.setdefault("me", None)
st.session_state.setdefault("matches", None)

st.title("🤝 AI Peer Learning Matcher")
st.caption("Find classmates who are strong where you're weak, and the other way round.")

models = ollama_models()
with st.sidebar:
    st.header("Status")
    if models is None:
        st.error("Ollama is not running. Open the Ollama app, then refresh.")
        model = None
    elif not models:
        st.warning("Ollama is running but has no models yet.")
        model = None
    else:
        st.success("Ollama is running")
        model = st.selectbox("Chat model", models)
    st.write("Students in pool:", len(st.session_state.students))

tab1, tab2, tab3, tab4 = st.tabs(["👤 My Profile", "🔍 Matches", "💬 Chatbot", "📋 All Students"])

with tab1:
    with st.form("profile"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name")
        branch = c2.text_input("Branch", "CSE")
        strong = st.text_input("Subjects I'm strong in (comma separated)", "Python, Data Structures")
        need = st.text_input("Subjects I need help in (comma separated)", "Engineering Chemistry")
        c3, c4 = st.columns(2)
        avail = c3.selectbox("Availability", ["Mornings", "Evenings", "Weekends"])
        style = c4.selectbox("Study style", ["Problem solving", "Step-by-step", "Discussion",
                                             "Practice tests", "Visual diagrams", "Project-based"])
        go = st.form_submit_button("Find my matches")
    if go:
        if not name.strip():
            st.error("Please enter your name.")
        else:
            me = dict(name=name.strip(), branch=branch, strong_in=strong,
                      need_help_in=need, availability=avail, style=style)
            st.session_state.me = me
            df = st.session_state.students
            df = pd.concat([df[df["name"].str.lower() != me["name"].lower()],
                            pd.DataFrame([me])], ignore_index=True)
            st.session_state.students = df
            df.to_csv(CSV, index=False)
            with st.spinner("Matching..."):
                st.session_state.matches = find_matches(me, df)
            st.success("Done! Open the Matches tab.")

with tab2:
    m = st.session_state.matches
    if m is None:
        st.info("Create your profile first.")
    else:
        for _, p in m.iterrows():
            with st.container(border=True):
                a, b = st.columns([3, 1])
                a.subheader(f"{p['name']} ({p['branch']})")
                a.write(f"**Can teach you:** {p['strong_in']}")
                a.write(f"**Needs help in:** {p['need_help_in']}")
                a.write(f"🕒 {p['availability']}  |  📚 {p['style']}")
                b.metric("Match", f"{p['match_score']}%")
                b.progress(min(int(p["match_score"]), 100))
        st.bar_chart(m.set_index("name")[["they_help_me", "i_help_them"]])

with tab3:
    st.write("Ask for a study plan, icebreakers, or explanations.")
    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("e.g. Make a 1-week study plan with my top match"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        ctx = ""
        if st.session_state.me:
            ctx += f"Student profile: {st.session_state.me}\n"
        mm = st.session_state.matches
        if mm is not None and len(mm):
            ctx += "Top matches:\n" + mm.head(3)[
                ["name", "strong_in", "need_help_in", "availability", "match_score"]].to_string(index=False)
        with st.chat_message("assistant"):
            if model is None:
                reply = "Ollama has no model available. Check the sidebar."
            else:
                with st.spinner("Thinking..."):
                    reply = ask(model, st.session_state.chat, ctx)
            st.write(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})

with tab4:
    st.dataframe(st.session_state.students, use_container_width=True)
