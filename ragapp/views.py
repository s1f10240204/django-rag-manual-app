import os
import requests
import uuid
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from .models import ProcessedManual
from .rag_handler import create_vectorstore_from_vision_pdf, ask_question
from googlesearch import search

SUGGESTED_DATA = {
    'aircon': {
        'name': '💨 エアコン', 'slug': 'aircon',
        'products': [
            {'name': '三菱電機 霧ヶ峰 MSZ-ZW4024S', 'icon': '💨'},
            {'name': 'ダイキン うるさらX AN40YRP', 'icon': '💨'},
            {'name': '日立 白くまくん RAS-X40N2', 'icon': '💨'},
            {'name': 'パナソニック エオリア CS-LX404D2', 'icon': '💨'},
        ]
    },
    'fan': {
        'name': '🍃 扇風機', 'slug': 'fan',
        'products': [
            {'name': 'バルミューダ The GreenFan EGF-1800', 'icon': '🍃'},
            {'name': 'ダイソン Purifier Hot+Cool HP10', 'icon': '🍃'},
            {'name': 'パナソニック F-CW339', 'icon': '🍃'},
            {'name': 'アイリスオーヤマ PCF-SC15T', 'icon': '🍃'},
        ]
    },
    'cleaner': {
        'name': '🧹 掃除機', 'slug': 'cleaner',
        'products': [
            {'name': 'Dyson V15 Detect', 'icon': '🧹'},
            {'name': 'iRobot Roomba Combo j9+', 'icon': '🧹'},
            {'name': 'Panasonic MC-NS100K', 'icon': '🧹'},
            {'name': 'Shark EVOPOWER SYSTEM iQ+', 'icon': '🧹'},
        ]
    },
    'tv': {
        'name': '📺 テレビ', 'slug': 'tv',
        'products': [
            {'name': 'Sony BRAVIA (ブラビア)', 'icon': '📺'},
            {'name': 'Panasonic VIERA (ビエラ)', 'icon': '📺'},
            {'name': 'Sharp AQUOS (アクオス)', 'icon': '📺'},
            {'name': 'LG OLED TV', 'icon': '📺'},
        ]
    },
    'camera': {
        'name': '📷 カメラ', 'slug': 'camera',
        'products': [
            {'name': 'Sony α7 IV', 'icon': '📷'},
            {'name': 'Canon EOS R6 Mark II', 'icon': '📷'},
            {'name': 'Nikon Z8', 'icon': '📷'},
            {'name': 'FUJIFILM X-T5', 'icon': '📷'},
        ]
    },
    'headphone': {
        'name': '🎧 オーディオ', 'slug': 'headphone',
        'products': [
            {'name': 'Sony WH-1000XM5', 'icon': '🎧'},
            {'name': 'Apple AirPods Pro 2', 'icon': '🎧'},
            {'name': 'Bose QuietComfort Ultra Headphones', 'icon': '🎧'},
            {'name': 'Anker Soundcore Liberty 4', 'icon': '🎧'},
        ]
    }
}

def load_manual_view(request):
    if request.method == 'GET':
        category_slug = request.GET.get('category')
        if category_slug and category_slug in SUGGESTED_DATA:
            category_info = SUGGESTED_DATA[category_slug]
            context = {'suggested_products': category_info['products'], 'current_category_name': category_info['name']}
            return render(request, 'ragapp/load_manual.html', context)
        else:
            context = {'categories': SUGGESTED_DATA.values()}
            return render(request, 'ragapp/load_manual.html', context)

    if request.method == 'POST':
        product_name_raw = request.POST.get('product_name', '').strip()
        current_category_slug = request.GET.get('category')
        def render_error(error_message):
            if current_category_slug and current_category_slug in SUGGESTED_DATA:
                category_info = SUGGESTED_DATA[current_category_slug]
                context = {'error': error_message, 'suggested_products': category_info['products'], 'current_category_name': category_info['name']}
                return render(request, 'ragapp/load_manual.html', context)
            else:
                context = {'error': error_message, 'categories': SUGGESTED_DATA.values()}
                return render(request, 'ragapp/load_manual.html', context)

        if not product_name_raw: return render_error('製品名を入力してください。')

        product_name = product_name_raw.lower()
        manual, created = ProcessedManual.objects.get_or_create(product_name=product_name)

        if not created and manual.status == 'COMPLETED':
            request.session['vectorstore_path'] = manual.vectorstore_path; request.session['product_name'] = product_name_raw
            return redirect('chat')
        
        if manual.status == 'FAILED': manual.status = 'COMPLETED'; manual.save()

        query = f'"{product_name_raw}" 取扱説明書 filetype:pdf'
        pdf_url = None
        try:
            for url in search(query, num_results=5, lang="ja", sleep_interval=1):
                if url.endswith('.pdf'): pdf_url = url; break
        except Exception as e: return render_error(f'Google検索中にエラーが発生: {e}')
        
        if not pdf_url: return render_error('取扱説明書のPDFが見つかりませんでした。')

        temp_pdf_path = ""
        try:
            response = requests.get(pdf_url, timeout=30, verify=False); response.raise_for_status()
            temp_dir = os.path.join(settings.BASE_DIR, 'temp_manuals'); os.makedirs(temp_dir, exist_ok=True)
            temp_pdf_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
            with open(temp_pdf_path, 'wb') as f: f.write(response.content)
            
            vectorstore_id = str(manual.id)
            vectorstore_path = os.path.join(settings.BASE_DIR, 'vectorstores', vectorstore_id)
            success = create_vectorstore_from_vision_pdf(temp_pdf_path, vectorstore_path)

            if success:
                manual.vectorstore_path = vectorstore_path; manual.status = 'COMPLETED'; manual.save()
                request.session['vectorstore_path'] = vectorstore_path; request.session['product_name'] = product_name_raw
                return redirect('chat')
            else:
                manual.status = 'FAILED'; manual.save()
                return render_error('PDFの解析に失敗しました。(Popplerはインストールされていますか？)')
        except requests.exceptions.RequestException as e:
            manual.status = 'FAILED'; manual.save()
            return render_error(f'PDFのダウンロードに失敗: {e}')
        finally:
            if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)

    return render(request, 'ragapp/load_manual.html', {'categories': SUGGESTED_DATA.values()})

# ★★★ ここからが新しく追加する関数 ★★★
def upload_manual_view(request):
    """
    アップロードされたPDFファイルを処理するビュー
    """
    if request.method == 'POST':
        pdf_file = request.FILES.get('pdf_file')
        
        def render_error(error_message):
            """エラー時に表示を正しく元に戻すためのヘルパー関数"""
            context = {'error': error_message, 'categories': SUGGESTED_DATA.values()}
            return render(request, 'ragapp/load_manual.html', context)

        if not pdf_file:
            return render_error('ファイルが選択されていません。')
        
        if not pdf_file.name.lower().endswith('.pdf'):
            return render_error('PDFファイルを選択してください。')

        temp_pdf_path = ""
        try:
            # アップロードされたファイルを一時的に保存
            temp_dir = os.path.join(settings.BASE_DIR, 'temp_manuals')
            os.makedirs(temp_dir, exist_ok=True)
            temp_pdf_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{pdf_file.name}")
            with open(temp_pdf_path, 'wb+') as f:
                for chunk in pdf_file.chunks():
                    f.write(chunk)
            
            vectorstore_id = str(uuid.uuid4())
            vectorstore_path = os.path.join(settings.BASE_DIR, 'vectorstores', vectorstore_id)
            
            # PDFを解析してベクトルストアを作成
            success = create_vectorstore_from_vision_pdf(temp_pdf_path, vectorstore_path)

            if success:
                # 成功したらセッションに情報を保存してチャットページへ
                request.session['vectorstore_path'] = vectorstore_path
                request.session['product_name'] = f"アップロードされたファイル: {pdf_file.name}"
                return redirect('chat')
            else:
                return render_error('PDFの解析に失敗しました。(Popplerはインストールされていますか？)')
        finally:
            # 処理が終わったら一時ファイルを削除
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    # POSTリクエスト以外はトップページに戻す
    return redirect('load_manual')
# ★★★ ここまでが新しく追加する関数 ★★★

def chat_view(request):
    if not request.session.get('vectorstore_path'): return redirect('load_manual')
    context = {'product_name': request.session.get('product_name', 'マニュアル')}
    return render(request, 'ragapp/chat.html', context)

@csrf_exempt
@require_POST
def chat_api_view(request):
    vectorstore_path = request.session.get('vectorstore_path')
    if not vectorstore_path: return JsonResponse({'error': 'Session expired'}, status=400)
    question = request.POST.get('question', '')
    if not question: return JsonResponse({'error': 'Question is empty'}, status=400)
    answer = ask_question(question, vectorstore_path)
    return JsonResponse({'answer': answer})