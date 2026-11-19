"""把 src/ 底下所有子資料夾加進 import 路徑，讓各腳本可以維持「from cv import ...」這種扁平寫法。

資料夾只是用途分類（見 CLAUDE.md「程式碼結構」），模組名稱全部不重複，所以不需要改任何 import。
每支腳本一開頭先把 src/ 加進 sys.path，再 import 這個模組；腳本與測試都從 repo 根目錄執行，
data/、results/、figures/ 的相對路徑不受影響。
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
for _root, _dirs, _ in os.walk(_SRC):
    _dirs[:] = sorted(d for d in _dirs if d != "__pycache__")
    if _root not in sys.path:
        sys.path.insert(0, _root)      # insert(0)：我們自己的模組優先於 site-packages 裡的同名套件
