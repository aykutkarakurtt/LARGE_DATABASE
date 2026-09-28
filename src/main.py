import os
import time

import click

from chunker import create_chunks
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    TEST_DATA_DIR,
    VECTORSTORE_DIR,
)
from document_loader import discover_documents, load_document
from search_engine import SearchEngine


@click.command()
@click.option("--ingest", is_flag=True, help="Belgeleri indeksle")
@click.option("--data-dir", default=DATA_DIR, help="Veri dizini yolu")
@click.option("--chunk-size", default=CHUNK_SIZE, help="Chunk boyutu")
@click.option("--chunk-overlap", default=CHUNK_OVERLAP, help="Chunk overlap")
@click.option("--test", is_flag=True, help="Test verisi ile dene")
@click.option("--ui", is_flag=True, help="Gradio arayüzü aç")
def main(ingest, data_dir, chunk_size, chunk_overlap, test, ui):
    data_path = TEST_DATA_DIR if test else data_dir

    if ingest:
        ingest_documents(data_path, chunk_size, chunk_overlap)

    if ui:
        from app import launch_ui

        print("🚀 Gradio arayüz başlıyor...")
        print(f"📁 Veri dizini: {data_path}")
        launch_ui()
        return

    if not ingest:
        print(f"📁 Veri dizini: {data_path}")
        documents = discover_documents(data_path)
        if not documents:
            print("❌ Hiç belge bulunamadı.")
            print(
                f"   Lütfen {data_path} klasörüne PDF, TXT veya DOCX "
                "dosyaları ekleyin."
            )
            return
        print(f"📄 {len(documents)} belge bulundu:")
        for document_path in documents:
            print(f"   - {os.path.basename(document_path)}")
        print("\n💡 İndekslemek için --ingest bayrağını ekleyin.")


def ingest_documents(data_path, chunk_size, chunk_overlap):
    print(f"📁 Veri dizini: {data_path}")
    documents = discover_documents(data_path)
    if not documents:
        raise click.ClickException(
            f"Hiç belge bulunamadı. {data_path} klasörüne PDF, TXT veya "
            "DOCX ekleyin."
        )

    print(f"📄 {len(documents)} belge bulundu:")
    for document_path in documents:
        print(f"   - {os.path.basename(document_path)}")

    print("\n🔄 İndeksleniyor...")
    started_at = time.time()
    search_engine = SearchEngine()
    search_engine.vector_store.reset()
    total_chunks = 0
    indexed_documents = 0
    skipped_documents = 0

    for document_path in documents:
        try:
            document = load_document(document_path)
            chunks = create_chunks(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as error:
            skipped_documents += 1
            print(f"   ⚠️ {os.path.basename(document_path)} atlandı: {error}")
            continue

        if not chunks:
            skipped_documents += 1
            print(f"   ⚠️ {os.path.basename(document_path)} metin içermiyor")
            continue

        search_engine.ingest(chunks, save=False)
        indexed_documents += 1
        total_chunks += len(chunks)
        print(f"   ✅ {os.path.basename(document_path)} → {len(chunks)} parça")

    if total_chunks == 0:
        raise click.ClickException(
            "Hiçbir belge işlenemedi; mevcut indeks korundu ve kaydedilmedi. "
            f"Atlanan belge: {skipped_documents}. Yukarıdaki hata mesajlarına bakın."
        )

    search_engine.vector_store.save()
    elapsed = time.time() - started_at
    print("\n🎯 İndeksleme tamamlandı!")
    print(f"   📄 Toplam belge: {indexed_documents}")
    print(f"   📦 Toplam parça: {total_chunks}")
    print(f"   ⚠️ Atlanan belge: {skipped_documents}")
    print(f"   ⏱️ Süre: {elapsed:.2f} saniye")
    print(f"   💾 Veritabanı: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
