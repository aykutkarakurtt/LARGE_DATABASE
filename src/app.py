import gradio as gr

from config import (
    DEFAULT_TOP_K,
    SERVER_NAME,
    SERVER_PORT,
    SIMILARITY_THRESHOLD,
)
from document_loader import location_kind
from rag_service import RagService
from search_engine import SearchEngine

search_engine = SearchEngine()
rag_service = RagService(search_engine)


def _retrieve(query, top_k, threshold):
    query = (query or "").strip()
    if not query:
        return query, [], "Lütfen bir sorgu yazın."
    if search_engine.vector_store.is_empty():
        return query, [], "⚠️ Veritabanı boş. Önce belgeleri indeksleyin."

    try:
        top_k = max(1, int(top_k))
        threshold = float(threshold)
    except (TypeError, ValueError):
        return query, [], "Sonuç sayısı ve benzerlik eşiği geçerli olmalı."

    results = search_engine.search(query, top_k=top_k, threshold=threshold)
    if not results:
        return query, [], "Sonuç bulunamadı. Benzerlik eşiğini düşürmeyi deneyin."

    return query, results, None


def _format_results(results):
    return [
        [
            result["source"],
            f"{location_kind(result['source'])} {result['page']}",
            f"{result['score']:.4f}",
            result["text"],
        ]
        for result in results
    ]


def rag_fn(query, top_k, threshold, generate_answer=True):
    query, results, error = _retrieve(query, top_k, threshold)
    if error:
        return "", None, error

    table_data = _format_results(results)
    if not generate_answer:
        return (
            "LLM cevabı devre dışı; yalnızca arama sonuçları gösteriliyor.",
            table_data,
            f"✅ {len(results)} sonuç bulundu.",
        )

    rag_result = rag_service.answer_from_hits(query, results)
    if rag_result["warning"]:
        status = "⚠️ Kaynaklar bulundu, ancak yerel LLM cevap üretemedi."
    else:
        status = f"✅ RAG cevabı oluşturuldu; {len(results)} kaynak kullanıldı."

    return rag_result["answer"], table_data, status


def build_ui():
    with gr.Blocks(title="SemanticDoc Search") as demo:
        gr.Markdown("# 🔍 SemanticDoc — Semantik Belge Arama")
        gr.Markdown(
            "Büyük veri tabanınızdan metin, paragraf veya kelime ile "
            "anlamca benzer belge parçalarını bulun."
        )

        query_box = gr.Textbox(
            lines=3,
            placeholder="Sorgunuzu yazın...",
            label="Arama Sorgusu",
        )
        top_k_slider = gr.Slider(
            1,
            20,
            DEFAULT_TOP_K,
            step=1,
            label="Sonuç Sayısı",
        )
        threshold_slider = gr.Slider(
            0.0,
            1.0,
            SIMILARITY_THRESHOLD,
            step=0.01,
            label="Benzerlik Eşiği",
        )

        generate_answer_checkbox = gr.Checkbox(
            label="Yerel LLM ile cevap üret",
            value=True,
        )
        search_btn = gr.Button("Ara ve Cevapla", variant="primary")
        status = gr.Markdown()
        answer_box = gr.Textbox(
            lines=8,
            label="RAG Cevabı",
            placeholder="LLM cevabı burada görüntülenecek.",
            interactive=False,
        )
        result_table = gr.Dataframe(
            headers=["Belge", "Sayfa", "Skor", "Metin Örneği"],
            datatype=["str", "str", "str", "str"],
            label="Sonuçlar",
            wrap=True,
            line_breaks=True,
            column_widths=[320, 110, 100, 760],
            max_chars=2000,
            interactive=False,
            max_height=700,
        )

        search_btn.click(
            fn=rag_fn,
            inputs=[
                query_box,
                top_k_slider,
                threshold_slider,
                generate_answer_checkbox,
            ],
            outputs=[answer_box, result_table, status],
        )
        query_box.submit(
            fn=rag_fn,
            inputs=[
                query_box,
                top_k_slider,
                threshold_slider,
                generate_answer_checkbox,
            ],
            outputs=[answer_box, result_table, status],
        )

    return demo


def launch_ui():
    demo = build_ui()
    demo.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=False,
    )


if __name__ == "__main__":
    launch_ui()
