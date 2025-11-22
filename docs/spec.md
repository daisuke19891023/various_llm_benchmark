# various-llm-benchmark 仕様

このプロジェクトは、複数のLLMおよびエージェントフレームワークのCLIベースの実行環境を提供することを目的としています。Typerによるコマンドラインインターフェースを中心に、プロバイダーやエージェントの実装を追加しやすい構造を採用しています。

## 目的
- OpenAI・Anthropicなど複数LLMプロバイダーのシンプルな応答・対話履歴付き応答をCLIから呼び出せること（OpenAIはResponses APIを使用）。
- LangChain・Agno・OpenAI Agents SDKなどのエージェント呼び出しを同じCLI体系で拡張できること。
- 設定値は`.env`経由で`pydantic-settings`から読み込むこと。
- プロバイダーごとに`interfaces/commands`配下にコマンドを切り出し、`llm/providers`や`agents/providers`配下に実装を配置すること。

## ディレクトリ構成
```
src/various_llm_benchmark/
├─ settings.py                # 環境変数の読み込み
├─ interfaces/
│  ├─ cli.py                  # Typerエントリポイント
│  └─ commands/               # コマンドごとの実装
│     ├─ openai.py
│     └─ claude.py
├─ llm/
│  ├─ protocol.py             # LLM共通インターフェース
│  └─ providers/
│     ├─ openai/client.py
│     └─ anthropic/client.py
└─ agents/
   └─ providers/              # LangChain/Agno/OpenAI Agents SDKなどを将来追加
```

## CLI方針
- Typerで単一のエントリポイント`various-llm-benchmark`を提供。
- プロバイダーごとにサブコマンドを追加し、`prompt`による単発出力と`--history`による対話履歴をサポート。
- 応答は標準出力にテキストとして返し、今後の機能追加に備えて`LLMResponse`モデルにメタデータを格納。

## 設定
- `settings.Settings`で`.env`を読み込み、APIキーやデフォルトモデル、温度を管理。
- 必須キー: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- 任意キー: `OPENAI_MODEL`, `ANTHROPIC_MODEL`, `DEFAULT_TEMPERATURE`

## テスト方針
- LLM呼び出しはすべてモック化し、ネットワークI/Oを伴わない単体テストを作成。
- `pytest`で`tests/unit`配下に、`src`と同じ構造をミラーリングしてテストを配置。

## 今後の拡張
- `agents/providers`配下にLangChain・Agno・OpenAI Agents SDKの実装を追加。
- `interfaces/commands`に対応するエージェント用サブコマンドを追加し、対話フローを統一。
- `docs/reference/`配下に各コマンドの詳細仕様と例を蓄積。
