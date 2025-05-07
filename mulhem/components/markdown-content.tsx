"use client"

import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import rehypePrettyCode from 'rehype-pretty-code'
import rehypeRaw from 'rehype-raw'
import rehypeStringify from 'rehype-stringify'
import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

interface MarkdownContentProps {
  content: string
  className?: string
  dir?: 'rtl' | 'ltr'
}

export function MarkdownContent({ content, className, dir }: MarkdownContentProps) {
  const [html, setHtml] = useState('')
  
  // Auto-detect Arabic content
  const hasArabic = /[\u0600-\u06FF]/.test(content)
  const contentDir = dir || (hasArabic ? 'rtl' : 'ltr')

  useEffect(() => {
    const processMarkdown = async () => {
      const file = await unified()
        .use(remarkParse)
        .use(remarkRehype, { allowDangerousHtml: true })
        .use(rehypeRaw)
        .use(rehypePrettyCode, {
          theme: 'github-dark',
          keepBackground: true,
          defaultLang: 'plaintext',
        })
        .use(rehypeStringify)
        .process(content)

      setHtml(String(file))
    }

    processMarkdown()
  }, [content])

  return (
    <div 
      className={cn(
        'prose prose-slate dark:prose-invert max-w-none',
        'prose-headings:mb-4 prose-headings:mt-6',
        'prose-p:mb-4 prose-p:leading-7',
        'prose-pre:p-4 prose-pre:rounded-lg prose-pre:bg-slate-900',
        'prose-code:text-blue-500 dark:prose-code:text-blue-400',
        'prose-blockquote:border-l-4 prose-blockquote:border-slate-300 dark:prose-blockquote:border-slate-700',
        'prose-blockquote:pl-4 prose-blockquote:my-4',
        'prose-ul:list-disc prose-ul:pl-5 prose-ul:my-4',
        'prose-ol:list-decimal prose-ol:pl-5 prose-ol:my-4',
        hasArabic ? 'font-arabic text-right' : '',
        className
      )}
      dir={contentDir}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
