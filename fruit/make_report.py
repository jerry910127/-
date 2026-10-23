"""讀取真實訓練成果，產生中文修改說明 PDF。"""
import json
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors

# (字型路徑, ttc 子字型編號)；reportlab 只吃 TrueType 外框，Noto CJK 這類 CFF 字型無法使用。
FONT_CANDIDATES = [
    ('C:/Windows/Fonts/msjh.ttc', 0),                                # Windows 微軟正黑體
    ('/System/Library/Fonts/Supplemental/Arial Unicode.ttf', None),  # macOS
    ('/Library/Fonts/Arial Unicode.ttf', None),
    ('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 0),            # Linux／Colab：apt install fonts-wqy-zenhei
]


def register_zh_font():
    """依序嘗試系統中文字型；都不能用時改用 reportlab 內建的繁中 CID 字型，任何系統都能產生 PDF。"""
    for path, index in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont('ZH', path, **({} if index is None else {'subfontIndex': index})))
                return 'ZH'
            except Exception:
                continue
    pdfmetrics.registerFont(UnicodeCIDFont('MSung-Light'))
    return 'MSung-Light'


def main():
    root = Path(__file__).resolve().parent
    r = json.loads((root/'result/metrics.json').read_text(encoding='utf-8'))
    font = register_zh_font()
    body = ParagraphStyle('body', fontName=font, fontSize=10, leading=17,
                          spaceAfter=9, wordWrap='CJK')
    title = ParagraphStyle('title', parent=body, fontSize=22, leading=30,
                           textColor=colors.HexColor('#155e63'), spaceAfter=18)
    heading = ParagraphStyle('heading', parent=body, fontSize=14, leading=22,
                             textColor=colors.HexColor('#155e63'), spaceBefore=12)
    story = []
    def p(text, style=body): story.append(Paragraph(text, style))
    def table(rows, widths):
        t = Table([[Paragraph(str(v), body) for v in row] for row in rows], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0f2f1')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#b7cecd')),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7)]))
        story.append(t)
    p('水果影像 CNN 期中專案', title)
    p('程式修改、設計理由與實測報告 | 2026-10-23')
    p('一、如何符合公告', heading)
    p('依期中專案公告，完成可執行的個人影像分類專案，涵蓋資料處理、模型訓練及結果評估。資料改為 fruit.zip 中的水果影像，不沿用課堂 Fashion-MNIST 或溫度範例資料。任務是辨識蘋果、香蕉、橘子及各自的新鮮／腐爛外觀，共六類。')
    p('第 8 週報告時，現場以保存的最佳模型執行 predict.py 推論單張照片並說明程式碼；需要時也可用 train.py 重新訓練。')
    p('二、原註解與檔案保存方式', heading)
    p('根目錄既有程式不修改。fruit/original/ 完整保存課堂原始 cnn.py、dataset.py、test2.py，包含全部註解；fruit/previous/ 完整保存先前根目錄 LEGO 版本的三個程式。水果版在 fruit/ 建立獨立模組，避免原有雙 main、額外訓練迴圈及不同任務混用。')
    p('三、實際資料稽核', heading)
    m = r['metadata']
    p(f'壓縮檔有 dataset/ 與 dataset/dataset/ 兩套結構。本次僅採 dataset/train/ 與 dataset/test/，共 {m["canonical_entries"]:,} 張候選圖；依原始檔案 SHA-256 排除 {m["duplicates_removed"]:,} 張完全重複圖，跳過 {len(m["skipped"])} 張無法讀取圖，保留 {m["unique_images"]:,} 張。壓縮檔不解壓、不修改，RGB 縮圖載入記憶體供各回合使用。')
    rows = [['類別', '訓練', '驗證', '測試']]
    for c in m['counts']['train']:
        rows.append([c] + [m['counts'][s][c] for s in ('train', 'val', 'test')])
    rows.append(['總計'] + [sum(m['counts'][s].values()) for s in ('train', 'val', 'test')])
    table(rows, [208, 87, 87, 87])
    story.append(PageBreak())
    p('改了什麼，以及為什麼', title)
    table([['檔案', '修改與設計理由'],
        ['dataset.py', '由 ZIP 讀取六類影像；捨棄重複資料夾，移除完全相同內容。保留原 test，只從 train 逐類抽 15% 作驗證，種子 42。輸出 manifest 與稽核 JSON，讓每張圖的去向可追查。'],
        ['cnn.py', '影像輸入改為 RGB 三通道 64×64；三組 16/32/64 通道卷積搭配 BatchNorm、ReLU 與池化。用自適應池化取代固定 28×28 尺寸假設，Dropout 0.3 降低過擬合，輸出六類 logits。'],
        ['train.py', '單一入口完成讀取、訓練、驗證與最終測試。CrossEntropy 直接接 logits，Adam 學習率 0.001、weight decay 0.0001。僅依驗證 loss 保存最佳權重，連續五回合未改善即停止，避免測試集參與調參。'],
        ['評估輸出', '加入訓練／驗證 loss 與 accuracy 曲線、測試 accuracy、macro F1、逐類 precision/recall/F1、混淆矩陣及逐張預測。另報告永遠猜訓練最多類別的基準，確認模型是否優於簡單猜測。'],
        ['predict.py', '讀取保存的類別名稱、尺寸與正規化設定，推論使用相同前處理。缺少模型直接報錯，不以隨機權重冒充訓練成果。'],
        ['README／requirements', '提供安裝、訓練、推論與報告重建指令；加入資料格式、限制與檔案保存說明。fruit.zip 超出一般 GitHub 檔案限制，僅本機保留，不提交。']], [92, 377])
    p('前處理與資料增強', heading)
    p('圖片校正 EXIF 方向、轉 RGB、縮至 64×64，像素轉 0~1 後以 (x-0.5)/0.5 映射到 -1~1。只在訓練時隨機水平翻轉，驗證及測試固定前處理；這裡使用固定尺度，沒有用全體影像估計統計量。CPU 使用四執行緒，固定亂數種子；CUDA 可用時自動選擇 GPU。')
    story.append(PageBreak())
    p('真實訓練與結果評估', title)
    p(f'本次以完整處理後資料訓練，實際執行 {r["epochs_run"]} 回合，最佳驗證權重在第 {r["best_epoch"]} 回合；裝置 {r["device"]}，包含資料讀取及訓練共 {r["elapsed_seconds"]/60:.1f} 分鐘。預設最多 15 回合（可用 --epochs 調整），驗證 loss 連續 5 回合未改善即提前停止；本報告數據對應這次實測。')
    table([['指標', '測試結果'], ['Accuracy', f'{r["test_accuracy"]:.2%}'],
        ['Macro F1', f'{r["macro_f1"]:.4f}'], ['多數類別基準 Accuracy', f'{r["majority_baseline_accuracy"]:.2%}'],
        ['Cross entropy', f'{r["test_loss"]:.4f}']], [240, 229])
    story.append(Spacer(1, 10))
    table([['類別', 'Precision', 'Recall', 'F1', '樣本數']] +
          [[v['class'], f'{v["precision"]:.3f}', f'{v["recall"]:.3f}', f'{v["f1"]:.3f}', v['support']] for v in r['per_class']], [169, 75, 75, 75, 75])
    story.append(Spacer(1, 14))
    story.append(Image(str(root/'result/learning_curves.png'), width=469, height=164))
    story.append(PageBreak())
    p('模型混淆與使用限制', title)
    story.append(Image(str(root/'result/confusion_matrix.png'), width=442, height=332))
    p('矩陣縱軸是真實類別，橫軸是預測類別；對角線為正確辨識，非對角線可看出模型混淆來源。Macro F1 對每類等權平均，避免大量類別主導整體準確率。')
    p('仍需誠實說明的限制', heading)
    p('SHA-256 只能排除位元完全相同的影像，不能辨識旋轉、裁切或翻轉的近似圖；資料集離線增強來源與拍攝個體沒有可靠群組標籤，因此隨機驗證可能偏樂觀。保留原 test 也不等於保證拍攝來源獨立。未宣稱所有資料洩漏已排除，建議另外拍攝一批獨立照片測試。')
    p('本模型只能在六個既定類別中選一個，未知水果、複雜背景、不同光源與不同相機都可能誤判；softmax 分數不是校準後的正確機率。分類代表外觀標籤，不能作為食品安全或可食用判斷。')
    p('展示與重現', heading)
    p('安裝：python -m pip install -r fruit/requirements.txt<br/>訓練：python fruit/train.py --epochs 15<br/>推論：python fruit/predict.py &quot;你的照片.jpg&quot;<br/>重建 PDF：python fruit/make_report.py')
    p('本次輸出保存在 fruit/result/；PDF 保存在 fruit/output/pdf/。資料集 ZIP 檔案過大未上傳 GitHub，重新訓練前需放在 fruit/fruit.zip。')
    dest = root/'output/pdf/fruit_changes_report.pdf'; dest.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setFont(font, 9); canvas.setFillColor(colors.HexColor('#526b6b'))
        canvas.drawString(42, 25, '水果 CNN | 修改說明與實測')
        canvas.drawRightString(553, 25, str(doc.page))
    SimpleDocTemplate(str(dest), pagesize=(595, 842), leftMargin=55, rightMargin=55,
        topMargin=42, bottomMargin=45).build(story, onFirstPage=footer, onLaterPages=footer)
    print(dest)


if __name__ == '__main__':
    main()
