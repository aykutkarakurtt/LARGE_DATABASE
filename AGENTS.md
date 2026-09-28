# AGENTS.md

PDF/TXT/DOCX belgelerinde Türkçe/İngilizce semantik arama yapan ve yerel Ollama ile kaynaklı RAG cevabı üreten Gradio uygulaması.

## Komutlar

- `python src/main.py --ingest` → `data/` içindeki belgeleri indeksler
- `python src/main.py --ingest --test` → `data/test/` içindeki belgeleri indeksler
- `python src/main.py --ui` → Gradio arayüzünü açar
- `python src/main.py` → belgeleri listeler
- `python -m unittest discover -s tests -v` → testleri çalıştırır
- `python -m compileall -q src tests` → Python dosyalarının sözdizimini doğrular

**Kritik:** `--ingest` her çalıştırmada `vector_store.reset()` (src/main.py:76) ile indeksi **sıfırdan** kurar; kısmi/eklemeli güncelleme yok.

## Mimari akış

`data/` → `document_loader.py` → `chunker.py` (500 char, 50 overlap) → `embedder.py` (e5-base, 768-boyut) → FAISS `IndexFlatIP` (`vectorstore/faiss_index.bin` + `metadata.json`) → `rag_service.py` → Ollama `/api/chat`.

## Kritik noktalar

- **e5 önekleri zorunlu:** `intfloat/multilingual-e5-base`, metinler için `passage: ` (`embedder.py:17`) ve sorgular için `query: ` (`embedder.py:27`) önekini gerektirir. Kaldırılırsa skorlar ciddi bozulur.
- **Model değişince indeksi sil:** `VectorStore.__init__` mevcut `vectorstore/*` dosyalarını otomatik yükler; boyut/model uyuşmazlığı hata fırlatır. Farklı model kullanırken önce `vectorstore/faiss_index.bin` ve `metadata.json` silinmeli.
- **Kalıcı indeksin kaynağı:** `vectorstore/` şu an `data/test/837977158-Loresima-Bu-lbu-l-Kapanı-1-m-N.pdf` içeriğiyle kurulu. `--ui` doğrudan bu indeksi sorgular; belge seti değiştiyse önce yeniden `--ingest`.
- **Port:** `SERVER_PORT=7861`, `SERVER_NAME=0.0.0.0` (LAN'dan erişilebilir), `share=False` (public link yok). UI tekrar tekrar başlatılırken port 7861 dolu hatası olursa eski süreci öldür.
- **UI dili:** Arayüz ve çıktı tablosu Türkçe (`Belge | Sayfa | Skor | Metin Örneği`).
- **Yerel LLM:** Varsayılan model `qwen3:4b-instruct`; `RagService` kaynakları `[1]`, `[2]` biçiminde numaralar.
- **LLM fallback'i:** Ollama hata verirse arama sonuçları korunur ve cevap alanında uyarı gösterilir.

## Ortam

- Python 3.13 (anaconda), macOS. Git repo değil; CI, test ve lint yapılandırması yok (`requirements.txt` tek bağımlılık kaynağı).