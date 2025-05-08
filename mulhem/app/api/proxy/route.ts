import { NextRequest, NextResponse } from 'next/server';

/**
 * معالج API للعمل كوسيط CORS
 * يتيح للواجهة الأمامية الاتصال بالخادم الخلفي بدون مشاكل CORS
 */
export async function POST(request: NextRequest) {
  try {
    // استخراج عنوان URL المستهدف من الطلب
    const body = await request.json();
    const { url, method = 'POST', headers = {}, body: requestBody } = body;

    if (!url) {
      return NextResponse.json(
        { error: 'URL is required' },
        { status: 400 }
      );
    }

    console.log(`Proxying request to: ${url}`);

    // تحويل الرؤوس إلى كائن قياسي
    const requestHeaders: HeadersInit = {};
    Object.entries(headers).forEach(([key, value]) => {
      if (typeof value === 'string') {
        requestHeaders[key] = value;
      }
    });

    // إرسال الطلب إلى الخادم المستهدف
    const response = await fetch(url, {
      method,
      headers: {
        ...requestHeaders,
        // إضافة رؤوس مهمة
        'Content-Type': headers['Content-Type'] || 'application/json',
      },
      body: requestBody ? JSON.stringify(requestBody) : undefined,
    });

    // قراءة البيانات من الاستجابة
    const data = await response.json().catch(() => null);

    // إرجاع الاستجابة مع رؤوس CORS المناسبة
    return NextResponse.json(data, {
      status: response.status,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    });
  } catch (error) {
    console.error('Proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to proxy request', details: (error as Error).message },
      { status: 500 }
    );
  }
}

/**
 * معالج طلبات OPTIONS للتعامل مع طلبات preflight CORS
 */
export async function OPTIONS() {
  return NextResponse.json({}, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
}

/**
 * معالج خاص لرفع الملفات
 * يستخدم لتمرير طلبات FormData
 */
export async function PUT(request: NextRequest) {
  try {
    // استخراج عنوان URL المستهدف من الطلب
    const formData = await request.formData();
    const targetUrl = formData.get('target_url') as string;
    
    if (!targetUrl) {
      return NextResponse.json(
        { error: 'target_url is required in form data' },
        { status: 400 }
      );
    }
    
    // إزالة حقل target_url من البيانات
    formData.delete('target_url');
    
    console.log(`Proxying file upload to: ${targetUrl}`);
    
    // إرسال الطلب إلى الخادم المستهدف
    const response = await fetch(targetUrl, {
      method: 'POST',
      body: formData,
    });
    
    // قراءة البيانات من الاستجابة
    const data = await response.json().catch(() => null);
    
    // إرجاع الاستجابة مع رؤوس CORS المناسبة
    return NextResponse.json(data, {
      status: response.status,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    });
  } catch (error) {
    console.error('File upload proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to proxy file upload', details: (error as Error).message },
      { status: 500 }
    );
  }
} 