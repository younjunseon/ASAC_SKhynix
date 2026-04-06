"""
eda.ipynb에서 모든 이미지(base64 PNG)를 추출하여 PNG 파일로 저장하는 스크립트.
"""
import json
import base64
import os
import re
import unicodedata

def sanitize_filename(text, max_len=80):
    """한글 포함 텍스트를 안전한 파일명으로 변환."""
    # 파일명에 사용 불가한 문자 제거
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    # 연속 공백/특수문자 정리
    text = re.sub(r'[\s]+', '_', text.strip())
    # 괄호, 쉼표 등 정리
    text = re.sub(r'[()（）,，+→]', '_', text)
    # 연속 언더스코어 정리
    text = re.sub(r'_+', '_', text)
    text = text.strip('_')
    # 길이 제한
    if len(text) > max_len:
        text = text[:max_len].rstrip('_')
    return text


def get_cell_title(cell):
    """셀 소스에서 제목(첫 줄 주석)을 추출."""
    source = ''.join(cell.get('source', []))
    first_line = source.split('\n')[0].strip()
    # '#' 주석 제거
    title = first_line.lstrip('#').strip()
    # 코드인 경우 변수명 등 그대로 사용
    return title if title else 'untitled'


def extract_images(notebook_path, output_dir):
    """노트북에서 모든 이미지를 추출하여 PNG로 저장."""
    os.makedirs(output_dir, exist_ok=True)

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    img_index = 0
    saved = []

    for cell_idx, cell in enumerate(nb['cells']):
        outputs = cell.get('outputs', [])
        img_in_cell = 0

        for output in outputs:
            if output.get('output_type') not in ('display_data', 'execute_result'):
                continue
            data = output.get('data', {})
            if 'image/png' not in data:
                continue

            # base64 데이터 추출
            b64_data = data['image/png']
            if isinstance(b64_data, list):
                b64_data = ''.join(b64_data)
            b64_data = b64_data.strip()

            # 파일명 생성
            title = get_cell_title(cell)
            safe_title = sanitize_filename(title)

            if img_in_cell == 0:
                filename = f"{img_index:02d}_cell{cell_idx:03d}_{safe_title}.png"
            else:
                filename = f"{img_index:02d}_cell{cell_idx:03d}_{safe_title}_{img_in_cell+1}.png"

            # PNG로 저장
            filepath = os.path.join(output_dir, filename)
            img_bytes = base64.b64decode(b64_data)
            with open(filepath, 'wb') as f:
                f.write(img_bytes)

            size_kb = len(img_bytes) / 1024
            saved.append((filename, size_kb))
            print(f"  [{img_index:2d}] {filename}  ({size_kb:.1f} KB)")

            img_index += 1
            img_in_cell += 1

    return saved


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(script_dir, 'eda.ipynb')
    output_dir = os.path.join(script_dir, 'images')

    print(f"노트북: {notebook_path}")
    print(f"출력 폴더: {output_dir}")
    print(f"{'='*60}")

    saved = extract_images(notebook_path, output_dir)

    print(f"{'='*60}")
    print(f"총 {len(saved)}개 이미지 추출 완료 → {output_dir}")
    total_mb = sum(kb for _, kb in saved) / 1024
    print(f"총 용량: {total_mb:.1f} MB")
