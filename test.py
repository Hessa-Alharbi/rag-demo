import requests
import os
import sys
import json
from getpass import getpass

# الإعدادات
API_URL = "https://4161-2a02-cb80-4271-93aa-dc2c-9eea-2d8e-7325.ngrok-free.app/initialize"

def get_token():
    """الحصول على توكن المصادقة إما من البيئة أو من المستخدم"""
    token = os.environ.get("AUTH_TOKEN")
    if not token:
        print("لم يتم العثور على توكن في متغيرات البيئة.")
        token = getpass("أدخل توكن المصادقة (سيكون مخفياً): ")
    return token

def upload_file(file_path, token):
    """رفع ملف إلى الباكند"""
    # إزالة علامات الاقتباس إذا وجدت
    file_path = file_path.strip('"\'')
    
    if not os.path.exists(file_path):
        print(f"خطأ: الملف '{file_path}' غير موجود.")
        return None
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    print(f"جاري رفع الملف: {file_path}...")
    
    try:
        with open(file_path, "rb") as file:
            files = {"file": (os.path.basename(file_path), file)}
            response = requests.post(API_URL, headers=headers, files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("استجابة الخادم:", json.dumps(data, indent=2, ensure_ascii=False))
            
            session_id = data.get("session_id") or data.get("task_id")
            print(f"\nتم رفع الملف بنجاح!")
            print(f"رابط الواجهة: /chat/{session_id}")
            print(f"الرابط الكامل: https://f3c5-2a02-cb80-4271-93aa-dc2c-9eea-2d8e-7325.ngrok-free.app/chat/{session_id}")
            return session_id
        else:
            print(f"فشل في رفع الملف (كود الاستجابة: {response.status_code})")
            print(f"الرسالة: {response.text}")
            return None
    except Exception as e:
        print(f"حدث خطأ أثناء رفع الملف: {str(e)}")
        return None

def main():
    """الدالة الرئيسية"""
    print("==== أداة رفع الملفات إلى الباكند ====")
    
    # الحصول على مسار الملف من المستخدم أو من وسيطات سطر الأوامر
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("أدخل المسار الكامل للملف (بدون علامات اقتباس): ")
    
    # الحصول على توكن المصادقة
    token = get_token()
    
    # تحقق من عنوان API
    global API_URL
    custom_api = input(f"عنوان API (اضغط Enter للاستخدام الافتراضي {API_URL}): ")
    if custom_api:
        API_URL = custom_api
    
    # رفع الملف
    session_id = upload_file(file_path, token)
    
    if session_id:
        print("\nيمكن الآن استخدام واجهة المستخدم لطرح الأسئلة حول هذا المستند.")

if __name__ == "__main__":
    main()
