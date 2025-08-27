from django.db import models

class ProcessedManual(models.Model):
    """
    処理済みの取扱説明書の情報を格納するモデル
    """
    product_name = models.CharField(max_length=255, unique=True, db_index=True)
    vectorstore_path = models.CharField(max_length=512, blank=True)
    
    STATUS_CHOICES = [
        ('COMPLETED', '完了'),
        ('FAILED', '失敗'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='COMPLETED')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.product_name} ({self.get_status_display()})"