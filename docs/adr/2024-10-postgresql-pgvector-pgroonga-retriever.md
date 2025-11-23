# ADR: PostgreSQL + pgvector + PGroongaによるretriever設計

- 日付: 2024-10-13
- ステータス: Accepted

## 背景
DB連携のリトリーバーでは、Embedding検索と全文検索の両方を安定的に扱う必要がある。LLM実験用のサンドボックスであるため、ローカル環境で完結しつつ、クラウド移行も容易な構成を選びたい。

## 選択肢
- **PostgreSQL + pgvector + PGroonga (採用)**: 単一RDBでベクトル検索と全文検索を同時に扱える。拡張導入のみで済み、スキーマ設計も共有しやすい。
- Supabase pgvector: マネージドで運用負荷は低いが、PGroonga併用が難しく、ネットワーク経由の遅延が増える。
- 専用ベクターストア (Pinecone、Chromaなど): ベクトル専用で性能は良いが、全文検索を別サービスに分ける必要があり、運用が分散する。
- Elasticsearch/OpenSearchのみ: 全文検索は得意だがベクトル性能やモデル更新の柔軟性に制約がある。Python依存関係も追加で増える。

## 決定
PostgreSQLをベースに、pgvectorでEmbedding類似検索、PGroongaで全文検索を行う構成を採用する。単一の接続文字列・スキーマで両方のツールを切り替えられるようにし、環境変数で拡張の有効化やテーブル名を指定する。Dockerfile / docker-compose.ymlで拡張入りのPostgreSQLを起動できるようにし、ローカルとCIの再現性を確保する。

## 影響
- 環境変数: `POSTGRES_CONNECTION_STRING`、`POSTGRES_SCHEMA`、`EMBEDDING_MODEL`を必須とし、`ENABLE_PGVECTOR`/`ENABLE_PGROONGA`と各テーブル名で機能を切り替える。
- セットアップ: docker composeで拡張入りPostgreSQLを起動し、必要に応じてスキーマやテーブルを初期化する。`docker/initdb.d`配下に拡張有効化スクリプトを配置し、自動作成する。
- テスト/運用: 単一RDB構成のためマイグレーション管理がシンプルになる。パフォーマンス要件に応じて接続プールやインデックスの最適化を追加検討する。
