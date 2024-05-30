# -*- coding: utf-8 -*-
import asyncio
import subprocess
import tempfile

import streamlit as st
from markdown2 import markdown
from weasyprint import CSS, HTML

from graph.storm_graph import get_storm_graph


def on_copy_click(text):
    st.session_state.latest_copied_text = text
    process = subprocess.Popen(
        ["powershell.exe", "-command", "Set-Clipboard", "-Value", f'"{text}"'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # プロセスの完了を待つ
    _, stderr = process.communicate()

    # エラーが発生していないか確認
    if process.returncode != 0:
        print(f"Error: {stderr.decode('utf-8')}")
    else:
        st.success("クリップボードにコピーしました")


def create_pdf_from_markdown(markdown_text):
    html_text = markdown(markdown_text, extras=["fenced-code-blocks"])
    css = CSS(
        string="""
        @import url(https://fonts.googleapis.com/earlyaccess/notosansjp.css);
        /* 全体のフォント設定（日本語対応） */
        body {
            font-family: 'Noto Sans JP', sans-serif;
        }

        /* コードブロックのスタイル設定 */
        pre, code {
            font-family: 'Courier New', Courier, monospace;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow: auto;
        }

        /* PDF化を考慮したページの設定 */
        @media print {
            body {
                font-size: 12pt;
            }

            pre, code {
                font-size: 10pt;
                white-space: pre-wrap;
                word-wrap: break-word;
            }

            /* ページの余白設定 */
            @page {
                margin: 1in;
            }
        }

        """
    )

    # 一時ファイルに PDF を書き込む
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        HTML(string=html_text).write_pdf(temp_pdf.name, stylesheets=[css])
        return temp_pdf.name


async def generate_article(topic: str):
    storm = get_storm_graph()
    results = None
    async for step in storm.astream(
        {
            "topic": topic,
        }
    ):
        name = next(iter(step))
        print(name)
        print("-- ", str(step[name])[:300])
        results = step

    article = results["write_article"]["article"]
    return article


def input_topic_page():
    st.title("自動記事作成")
    st.info("作成したい記事のトピックを入力してください")
    topic = st.text_input("Topic")
    if st.button("実行"):
        if not topic:
            st.warning("トピックを入力してください")
        else:
            with st.spinner("実行中です..."):
                article = asyncio.run(generate_article(topic))
                st.session_state["article"] = article
                st.session_state["page_control"] = 1
                st.rerun()


def download_pdf_page():
    st.title("Result")
    container = st.container(border=True)
    article = st.session_state.get("article")

    if article:
        cols = container.columns([9, 1])
        with cols[0]:
            st.markdown(article)
        with cols[1]:
            st.button("📋", on_click=on_copy_click, args=(article,))
        # container.markdown(article)

        pdf_path = create_pdf_from_markdown(article)
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()

        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name="article.pdf",
            mime="application/pdf",
        )
    else:
        st.error("記事が作成できませんでした。もう一度試してください。")


def main():
    if st.session_state.get("page_control", 0) == 1:
        download_pdf_page()
    else:
        st.session_state["page_control"] = 0
        input_topic_page()


if __name__ == "__main__":
    main()
