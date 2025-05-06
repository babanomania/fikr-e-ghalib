import streamlit as st
from lib.agent_lib import ghalib_agent, summarize_article, extract_theme_sentiment, extract_roman_urdu_verse

st.title("🎙️ Fikr-e-Ghalib")
st.markdown("A poetic AI trained to channel the soul of Mirza Ghalib. "
           "Give it a news article URL, and it will summarize the event, extract its emotional essence, "
           "and compose a couplet in Roman Urdu—just as Ghalib might have reflected on it.")

url = st.text_input("Enter news article URL:")

if url:
    if not url.startswith("http"):
        st.error("❌ Please enter a valid news URL starting with http or https.")
    else:
        with st.spinner("📡 Reading and analyzing the news article..."):
            try:
                summary = summarize_article(url)
                theme, sentiment = extract_theme_sentiment(summary)
                verse = ghalib_agent(url)
                cleaned_verse = extract_roman_urdu_verse(verse)
                lines = [line.strip() for line in cleaned_verse.splitlines() if line.strip()]
                poetic_excerpt = "\n".join(lines[:2])

                st.subheader("📰 Summary")
                st.write(summary)

                st.subheader("🎭 Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Theme", theme.capitalize())
                with col2:
                    st.metric("Sentiment", sentiment.capitalize())

                st.subheader("✨ Ghalib-Style Verse")
                st.markdown(poetic_excerpt)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
