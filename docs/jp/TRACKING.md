# トラッキング

## 概要

このリポジトリのメイン関数は`run.py`です。これには引数パーサーが含まれており、検出、追跡、評価のそれぞれのサブ関数を呼び出します。`run.py`を呼び出す際に以下の引数を追加することで、それぞれのサブタスクを有効化または無効化できます。

- 検出: `--DETECTION True/False`
- 追跡: `--TRACKING True/False`
- 評価: `--EVALUATION True/False`

`--DETECTION True`を設定すると、選択された検出モデルが指定されたデータセット内のすべてのオブジェクトを検出し、検出結果を保存します。`--TRACKING True`は、利用可能な場合これらの検出結果を使用し、SAM2と上位のヒューリスティックを使用して検出されたオブジェクトを追跡します。`--EVALUATION True`は、TrackEvalパッケージを呼び出して追跡を評価します。

**警告:** `--DETECTION False`で保存された検出結果がない場合、モデルは追跡中に検出も行います。これは、適応的検出閾値を使用できないことを意味します。なぜなら、これらの閾値は最初に検出結果が利用可能である必要があるためです。

## 重要な引数

以下の引数はモデルを使用する際に有用です。これらは`run.py`でデフォルト値が指定されており、簡単に変更できます。

- `--DETECTION`: 追跡前にすべてのオブジェクトを検出（デフォルト: True）
- `--TRACKING`: オブジェクトを追跡（デフォルト: True）
- `--EVALUATION`: 追跡を評価（ground-truthsが必要）（デフォルト: True）

- `--TEST_NAME`: 実行の名前（デフォルト: test）
- `--SAVE_IMGS`: 予測結果を可視化した画像を保存（デフォルト: False）
- `--SAVE_VIDS`: 予測結果を含む保存された画像を動画に変換（デフォルト: False）

- `--LOAD_DETS`: 既存の検出結果を読み込む（検出結果が存在する場合、または同じデータセットで以前に`--DETECTION True`を使用した場合に有用）（デフォルト: False）

- `--DATA_SPLIT`: 使用するデータ分割（train/val/test）（デフォルト: val）
- `--DATA_ROOT`: すべてのデータセットの親ディレクトリへのパス（デフォルト: data）
- `--dataset`: 使用するデータセット

## 実行例

ChimpActのテスト分割ですべてのオブジェクトを追跡し、出力予測を可視化し、性能を評価したい場合:

`python tracker/run.py --dataset chimpact --DATA_SPLIT test --SAVE_IMGS True --DETECTION True`
