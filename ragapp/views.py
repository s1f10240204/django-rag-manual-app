import os
import requests
import uuid
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from .models import ProcessedManual
from .rag_handler import create_vectorstore_from_pdf, ask_question
from googlesearch import search

SUGGESTED_DATA = {
    'aircon': {
        'name': '💨 エアコン', 'slug': 'aircon',
        'products': [
            {'name': '三菱電機 霧ヶ峰 Zシリーズ', 'icon': '💨'},
            {'name': 'ダイキン うるさらX', 'icon': '💨'},
            {'name': '日立 白くまくん Xシリーズ', 'icon': '💨'},
            {'name': 'パナソニック エオリア LXシリーズ', 'icon': '💨'},
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

# load_manual_view 関数のみを置き換えます
def load_manual_view(request):
    if request.method == 'GET':
        category_slug = request.GET.get('category')
        if category_slug and category_slug in SUGGESTED_DATA:
            category_info = SUGGESTED_DATA[category_slug]
            context = {
                'suggested_products': category_info['products'],
                'current_category_name': category_info['name']
            }
            return render(request, 'ragapp/load_manual.html', context)
        else:
            context = {'categories': SUGGESTED_DATA.values()}
            return render(request, 'ragapp/load_manual.html', context)

    if request.method == 'POST':
        product_name_raw = request.POST.get('product_name', '').strip()
        # ★★★ エラー時に元の画面に戻るため、現在のカテゴリ情報を取得 ★★★
        current_category_slug = request.GET.get('category')

        def render_error(error_message):
            """エラー時に表示を正しく元に戻すためのヘルパー関数"""
            if current_category_slug and current_category_slug in SUGGESTED_DATA:
                # 製品一覧画面でエラーが起きた場合
                category_info = SUGGESTED_DATA[current_category_slug]
                context = {
                    'error': error_message,
                    'suggested_products': category_info['products'],
                    'current_category_name': category_info['name']
                }
                return render(request, 'ragapp/load_manual.html', context)
            else:
                # カテゴリ選択画面でエラーが起きた場合
                context = {'error': error_message, 'categories': SUGGESTED_DATA.values()}
                return render(request, 'ragapp/load_manual.html', context)

        if not product_name_raw:
            return render_error('製品名を入力してください。')

        product_name = product_name_raw.lower()
        manual, created = ProcessedManual.objects.get_or_create(product_name=product_name)

        if not created and manual.status == 'COMPLETED':
            request.session['vectorstore_path'] = manual.vectorstore_path; request.session['product_name'] = product_name_raw
            return redirect('chat')
        
        if manual.status == 'FAILED':
            manual.status = 'COMPLETED'; manual.save()

        query = f'"{product_name_raw}" 取扱説明書 filetype:pdf'
        pdf_url = None
        try:
            for url in search(query, num_results=5, lang="ja", sleep_interval=1):
                if url.endswith('.pdf'): pdf_url = url; break
        except Exception as e:
            return render_error(f'Google検索中にエラーが発生: {e}')
        
        if not pdf_url:
            return render_error('取扱説明書のPDFが見つかりませんでした。')

        temp_pdf_path = ""
        try:
            response = requests.get(pdf_url, timeout=30, verify=False); response.raise_for_status()
            temp_dir = os.path.join(settings.BASE_DIR, 'temp_manuals'); os.makedirs(temp_dir, exist_ok=True)
            temp_pdf_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
            with open(temp_pdf_path, 'wb') as f: f.write(response.content)
            
            vectorstore_id = str(manual.id)
            vectorstore_path = os.path.join(settings.BASE_DIR, 'vectorstores', vectorstore_id)
            
            success = create_vectorstore_from_pdf(temp_pdf_path, vectorstore_path)

            if success:
                manual.vectorstore_path = vectorstore_path; manual.status = 'COMPLETED'; manual.save()
                request.session['vectorstore_path'] = vectorstore_path; request.session['product_name'] = product_name_raw
                return redirect('chat')
            else:
                manual.status = 'FAILED'; manual.save()
                return render_error('PDFの解析に失敗しました。複雑なファイル形式である可能性があります。(Popplerはインストールされていますか？)')
        except requests.exceptions.RequestException as e:
            manual.status = 'FAILED'; manual.save()
            return render_error(f'PDFのダウンロードに失敗: {e}')
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    return render(request, 'ragapp/load_manual.html', {'categories': SUGGESTED_DATA.values()})
def chat_view(request):
    if not request.session.get('vectorstore_path'):
        return redirect('load_manual')
    context = {'product_name': request.session.get('product_name', 'マニュアル')}
    return render(request, 'ragapp/chat.html', context)

# ↓↓↓ この @csrf_exempt から始まる関数全体が正しく存在するか確認してください ↓↓↓
@csrf_exempt
@require_POST
def chat_api_view(request):
    """
    チャットの質問を受け取り、AIの回答をJSONで返すAPIビュー
    """
    vectorstore_path = request.session.get('vectorstore_path')
    if not vectorstore_path:
        return JsonResponse({'error': 'Session expired'}, status=400)

    question = request.POST.get('question', '')
    if not question:
        return JsonResponse({'error': 'Question is empty'}, status=400)

    answer = ask_question(question, vectorstore_path)
    return JsonResponse({'answer': answer})