# RAG Service Enhancements

## Overview
This project enhances the RAG (Retrieval-Augmented Generation) service by improving document processing and response generation, particularly for Arabic text.

## Changes Made

1. **Library Update**:
   - Replaced `fitz` (PyMuPDF) with `PyPDFLoader` from `langchain_community.document_loaders` for better PDF handling.

2. **Text Extraction**:
   - Updated the `extract_file_content` method to utilize `PyPDFLoader` for extracting text from PDF files.
   - The method now cleans and formats the extracted text for better readability.

3. **Response Formatting**:
   - Implemented `clean_and_format_text` to clean unwanted characters and format lists properly.
   - Enhanced `format_response` to ensure that responses are well-structured, especially for Arabic text.

4. **Preprocessing Automation**:
   - Automated the cleaning and formatting of text responses to ensure consistency and clarity.
   - Improved handling of Arabic text, including punctuation and spacing.

## Usage
To use the updated service, ensure that the necessary libraries are installed and that the service is properly initialized. The service can now handle PDF documents more effectively and generate cleaner, more readable responses.

## Future Improvements
- Further enhancements to the text extraction process for other file types.
- Additional support for more languages and text formats.