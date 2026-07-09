# Chatbot LLM+RAG — SEO Content Automation Engine

> **Lưu ý về repo này:** Đây là bản *source code* được đẩy lên từ một dự án lớn hơn (SEO content bot chạy production cho nhiều website/brand khác nhau). Các phần sau **không được đưa lên GitHub**: dữ liệu người dùng thực (`uploads/`, database `projects.db`), file ghi chú nội bộ, ảnh minh hoạ đã sinh cho các bài viết cụ thể, và mọi tên sản phẩm/khách hàng cụ thể (đã được thay bằng tên ví dụ chung chung như "Acme"). File này mô tả lại kiến trúc đầy đủ để người xem hiểu bức tranh tổng thể dù không thấy dữ liệu thật trong repo.

## 1. Giới thiệu

Đây là một **hệ thống tự động hoá viết nội dung SEO** kết hợp **RAG (Retrieval-Augmented Generation)** và **LLM (OpenAI GPT)**. Người dùng upload tài liệu nghiệp vụ (PDF/DOCX/TXT/MD) về sản phẩm/thương hiệu, hệ thống xây dựng knowledge base bằng embedding cục bộ, sau đó dùng pipeline nhiều tầng (SERP research → phân tích intent → xây cụm chủ đề → viết bài có căn cứ từ knowledge base → chèn internal link → chấm điểm chất lượng → human review → publish) để sinh ra bài viết chuẩn SEO, có trích dẫn đúng dữ liệu nội bộ thay vì AI "bịa" thông tin.

Điểm khác biệt so với chatbot hỏi-đáp thông thường: đây là một **agent pipeline nhiều bước** dùng RAG làm nguồn sự thật (grounding) cho từng phần nội dung được LLM sinh ra, có gate chất lượng và con người duyệt trước khi xuất bản — không phải hội thoại 1 lượt hỏi 1 lượt trả lời.

## 2. Kiến trúc RAG (Retrieval-Augmented Generation)

```
Upload tài liệu (PDF/DOCX/TXT/MD)
   → document_loader.py: trích xuất text, chia chunk 500–800 ký tự
   → embedding_engine.py: embed CỤC BỘ bằng sentence-transformers
                            (all-MiniLM-L6-v2, 384 chiều, chạy trên máy, KHÔNG tốn API)
   → lưu vector .npy + index JSON (uploads/knowledge_index.json)

Khi cần viết nội dung cho 1 heading/từ khoá:
   → embed câu truy vấn cục bộ (free)
   → retrieval.py: semantic search bằng cosine similarity trong vector đã lưu (free)
   → lấy top-3 chunk liên quan nhất
   → CHỈ 3 chunk này + tiêu đề được gửi lên OpenAI để sinh nội dung
     (giảm chi phí API tối đa — không gửi toàn bộ tài liệu mỗi lần gọi LLM)
```

Thành phần chính (`core/knowledge/`):

| File | Vai trò |
|---|---|
| `document_loader.py` | Trích xuất text từ PDF/DOCX/TXT/MD, chunk hoá, quản lý index tài liệu |
| `embedding_engine.py` | Sinh embedding cục bộ (sentence-transformers), lưu/đọc vector `.npy` |
| `retrieval.py` | Semantic search (cosine similarity) để lấy chunk liên quan nhất cho mỗi truy vấn |

## 3. Pipeline sinh nội dung (LLM orchestration)

`ArticleController` (`core/content/article_controller.py`) là orchestrator chính:

```
UI (dashboard)
   → PreValidationLayer      validate keyword/title/heading/API key trước khi tốn tiền LLM
   → ArticleGenerator        wrap ArticleBuilder — pipeline 17 phase (outline, EEAT, brand angle,
                              section writer, depth controller, truncation detector...)
   → QualityGate             chấm điểm chất lượng tự động (non-blocking) + review thủ công (human-in-the-loop)
   → Renderer                dọn HTML cuối cùng
   → Lưu DB (draft)          gsc_dashboard.py xử lý lưu + publish
```

`bot_seo.py` là CLI orchestrate 8 "engine" cho pipeline đầy đủ chạy theo batch:

```
SERP research → Intent mapping → Topic clustering → Brand angle
   → Content generation (EEAT-aware) → Internal linking → Quality gate → Rank monitoring
```

| Layer | File | Vai trò |
|---|---|---|
| SERP | `core/serp/` | Crawl + parse kết quả tìm kiếm Google (qua SerpAPI), phát hiện gap nội dung |
| Intent | `core/intent/` | Phân loại intent người dùng, phát hiện gap giữa nội dung hiện có và intent |
| Cluster | `core/cluster/` | Xây dựng topical authority / cụm chủ đề liên quan |
| Content | `core/content/` | 17-phase article builder: outline, heading, EEAT, brand angle, section writer, depth control, truncation detection, internal link injection, versioning |
| LLM | `core/llm/` | Wrapper OpenAI (`llm_client.py`), prompt templates, EEAT framework |
| Quality | `core/quality/` | Quality gate — human-in-the-loop review trước khi publish |
| Linking | `core/linking/` | Tự động hoá chèn internal link dựa trên retrieval + keyword matching |
| Monitoring | `core/monitoring/` | Theo dõi thứ hạng (rank) và cập nhật SERP sau khi publish |
| Analytics | `core/analytics/` | Phân tích hiệu suất bài viết đã publish |
| Publisher | `core/publisher.py` | Publish trực tiếp qua WordPress REST API / Shopify Admin API (kèm upload featured image) |

## 4. Tech stack

- **LLM**: OpenAI GPT (`openai` SDK) — sinh nội dung, cấu hình model/temperature qua `.env`
- **RAG / Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2, chạy local, miễn phí)
- **SERP data**: SerpAPI (`google-search-results`)
- **Database**: PostgreSQL qua SQLAlchemy ORM + Alembic migration
- **Backend/Dashboard**: Flask + Jinja2, deploy bằng Gunicorn
- **Document parsing**: `pdfplumber`, `python-docx`
- **Google Search Console**: `google-api-python-client` (theo dõi ranking, dữ liệu search performance)
- **Publish**: WordPress REST API, Shopify Admin API

## 5. Cấu trúc mã nguồn trong repo này

```
bot_seo.py            CLI entry point — orchestrate pipeline đầy đủ (8 engine)
gsc_dashboard.py       Flask app — dashboard quản lý project, viết bài, publish, xem GSC data
gsc_tool.py            Tool tương tác Google Search Console API
config.py              Config tập trung (pydantic), load từ .env
core/
  knowledge/           RAG: document loader, local embeddings, retrieval
  llm/                 LLM client (OpenAI wrapper), prompt templates, EEAT framework
  content/             17-phase article builder (outline → EEAT → sections → render)
  serp/, intent/, cluster/   Nghiên cứu từ khoá & xây dựng topical authority
  linking/             Tự động chèn internal link
  quality/             Quality gate (human-in-the-loop)
  monitoring/, analytics/    Theo dõi rank & hiệu suất sau publish
  publisher.py         Publish lên WordPress / Shopify
  database.py, models.py    SQLAlchemy ORM (PostgreSQL)
dashboard/             Frontend Flask (templates + static JS/CSS) cho dashboard quản trị
```

### Không có trong repo (do chứa dữ liệu thật/không liên quan đến kiến trúc)

| Phần | Lý do loại bỏ |
|---|---|
| `uploads/` | Tài liệu người dùng thật đã upload + vector embedding đã sinh |
| `projects.db` | Database SQLite chứa dữ liệu project/bài viết thực tế |
| `generated_images/` | Ảnh minh hoạ đã sinh cho các bài viết cụ thể |
| File ghi chú nội bộ (`.docx`) | Không liên quan đến source code |

## 6. Cách chạy (cần tự cấu hình `.env` — xem `.env.example`)

```bash
pip install -r requirements.txt

# Khởi tạo database
python bot_seo.py init-db

# Chạy pipeline đầy đủ cho 1 từ khoá
python bot_seo.py run "<từ khoá>" --clusters 7 --language vi

# Hoặc chỉ chạy từng bước
python bot_seo.py analyze "<từ khoá>"     # SERP analysis
python bot_seo.py cluster "<từ khoá>"     # Topic cluster
python bot_seo.py write "<từ khoá>"       # Sinh nội dung
python bot_seo.py review                  # Danh sách bài chờ duyệt

# Dashboard (Flask)
python gsc_dashboard.py
```

## 7. Tác giả

Đinh Đức Hiệp
