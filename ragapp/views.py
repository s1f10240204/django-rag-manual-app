import os
import requests
import uuid
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

# 必要なモデルと関数をインポート
from .models import ProcessedManual
from .rag_handler import create_vectorstore_from_pdf, ask_question
from googlesearch import search


def load_manual_view(request):
    # このビューは変更ありません
    if request.method == 'POST':
        product_name_raw = request.POST.get('product_name', '').strip()
        if not product_name_raw:
            return render(request, 'ragapp/load_manual.html', {'error': '製品名を入力してください。'})

        product_name = product_name_raw.lower()
        manual, created = ProcessedManual.objects.get_or_create(product_name=product_name)

        if not created and manual.status == 'COMPLETED':
            request.session['vectorstore_path'] = manual.vectorstore_path
            request.session['product_name'] = product_name_raw
            return redirect('chat')
        
        if manual.status == 'FAILED':
            manual.status = 'COMPLETED'
            manual.save()

        query = f'"{product_name_raw}" 取扱説明書 filetype:pdf'
        pdf_url = None
        try:
            for url in search(query, num_results=5, lang="ja", sleep_interval=1):
                if url.endswith('.pdf'): pdf_url = url; break
        except Exception as e:
            return render(request, 'ragapp/load_manual.html', {'error': f'Google検索中にエラーが発生: {e}'})
        
        if not pdf_url:
            return render(request, 'ragapp/load_manual.html', {'error': '取扱説明書のPDFが見つかりませんでした。'})

        temp_pdf_path = ""
        try:
            response = requests.get(pdf_url, timeout=30, verify=False)
            response.raise_for_status()
            temp_dir = os.path.join(settings.BASE_DIR, 'temp_manuals'); os.makedirs(temp_dir, exist_ok=True)
            temp_pdf_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
            with open(temp_pdf_path, 'wb') as f: f.write(response.content)

            vectorstore_id = str(manual.id)
            vectorstore_path = os.path.join(settings.BASE_DIR, 'vectorstores', vectorstore_id)
            
            success = create_vectorstore_from_pdf(temp_pdf_path, vectorstore_path)

            if success:
                manual.vectorstore_path = vectorstore_path
                manual.status = 'COMPLETED'
                manual.save()
                
                request.session['vectorstore_path'] = vectorstore_path
                request.session['product_name'] = product_name_raw
                return redirect('chat')
            else:
                manual.status = 'FAILED'; manual.save()
                return render(request, 'ragapp/load_manual.html', {'error': 'PDFの解析に失敗しました。'})

        except requests.exceptions.RequestException as e:
            manual.status = 'FAILED'; manual.save()
            return render(request, 'ragapp/load_manual.html', {'error': f'PDFのダウンロードに失敗: {e}'})
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    return render(request, 'ragapp/load_manual.html')

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ この関数が views.py に存在するか確認してください ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
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


def chat_view(request):
    """
    チャットページの初期表示を行うビュー
    """
    if not request.session.get('vectorstore_path'):
        return redirect('load_manual')

    context = {
        'product_name': request.session.get('product_name', 'マニュアル'),
    }
    return render(request, 'ragapp/chat.html', context)