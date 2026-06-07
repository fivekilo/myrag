import React, { useMemo, useState } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const TYPE_STYLES = {
  title: 'bg-blue-100 text-blue-700',
  table: 'bg-amber-100 text-amber-700',
  caption: 'bg-purple-100 text-purple-700',
  footnote: 'bg-rose-100 text-rose-700',
  reference_item: 'bg-emerald-100 text-emerald-700',
  abstract_body: 'bg-sky-100 text-sky-700',
  paragraph: 'bg-gray-100 text-gray-700'
};

const ParseFile = () => {
  const [file, setFile] = useState(null);
  const [loadingMethod, setLoadingMethod] = useState('pymupdf');
  const [parsingOption, setParsingOption] = useState('structured_blocks');
  const [parsedContent, setParsedContent] = useState(null);
  const [status, setStatus] = useState('');

  const blocks = parsedContent?.blocks || [];
  const legacyContent = parsedContent?.content || [];
  const diagnostics = parsedContent?.diagnostics || {};

  const titleCount = useMemo(
    () => blocks.filter((block) => block.type === 'title').length,
    [blocks]
  );

  const handleProcess = async () => {
    if (!file || !loadingMethod || !parsingOption) {
      setStatus('Please select all required options');
      return;
    }

    setStatus('Processing...');
    setParsedContent(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('loading_method', loadingMethod);
      formData.append('parsing_option', parsingOption);

      const response = await fetch(`${apiBaseUrl}/parse`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setParsedContent(data.parsed_content);
      setStatus('Processing completed successfully!');
    } catch (error) {
      console.error('Error:', error);
      setStatus(`Error: ${error.message}`);
    }
  };

  const renderStructuredBlock = (block) => {
    const badgeClass = TYPE_STYLES[block.type] || 'bg-gray-100 text-gray-700';

    return (
      <div key={block.block_id} className="p-4 border rounded bg-gray-50">
        <div className="flex items-center justify-between gap-3 mb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`px-2 py-1 rounded text-xs font-semibold ${badgeClass}`}>
              {block.type}
            </span>
            <span className="text-xs text-gray-500">Page {block.page}</span>
            {block.level !== undefined && (
              <span className="text-xs text-gray-500">Level {block.level}</span>
            )}
            {block.column_index !== undefined && (
              <span className="text-xs text-gray-500">Column {block.column_index + 1}</span>
            )}
          </div>
          {Array.isArray(block.section_path) && block.section_path.length > 0 && (
            <div className="text-xs text-gray-400 text-right">
              {block.section_path.join(' / ')}
            </div>
          )}
        </div>

        <div className="text-sm text-gray-700 whitespace-pre-wrap">{block.text}</div>

        {block.type === 'table' && Array.isArray(block.rows) && block.rows.length > 0 && (
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full text-xs border border-gray-200">
              <tbody>
                {block.rows.map((row, rowIndex) => (
                  <tr key={`${block.block_id}-row-${rowIndex}`} className="border-t border-gray-200">
                    {row.map((cell, cellIndex) => (
                      <td
                        key={`${block.block_id}-cell-${rowIndex}-${cellIndex}`}
                        className="px-2 py-1 border-r border-gray-200"
                      >
                        {cell || ''}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  };

  const renderLegacyItem = (item, idx) => (
    <div key={idx} className="p-3 border rounded bg-gray-50">
      <div className="font-medium text-sm text-gray-500 mb-1">
        {item.type} - Page {item.page}
      </div>
      {item.title && (
        <div className="font-bold text-gray-700 mb-2">
          {item.title}
        </div>
      )}
      <div className="text-sm text-gray-600 whitespace-pre-wrap">
        {item.content}
      </div>
    </div>
  );

  return (
    <div className="p-6">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6">检索增强生成工具</h1>
      <hr />
      <h2 className="text-2xl font-bold mb-6">文件解析</h2>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div>
              <label className="block text-sm font-medium mb-1">选择 PDF 文件</label>
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => setFile(e.target.files[0] || null)}
                className="block w-full border rounded px-3 py-2"
                required
              />
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium mb-1">加载工具</label>
              <select
                value={loadingMethod}
                onChange={(e) => setLoadingMethod(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="pymupdf">PyMuPDF</option>
                <option value="pypdf">PyPDF</option>
                <option value="unstructured">Unstructured</option>
                <option value="pdfplumber">PDF Plumber</option>
              </select>
            </div>

            <div className="mt-4">
              <label className="block text-sm font-medium mb-1">解析选项</label>
              <select
                value={parsingOption}
                onChange={(e) => setParsingOption(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="structured_blocks">Structured Blocks</option>
                <option value="all_text">All Text</option>
                <option value="by_pages">By Pages</option>
                <option value="by_titles">By Titles</option>
                <option value="text_and_tables">Text and Tables</option>
              </select>
            </div>

            <button
              onClick={handleProcess}
              className="mt-4 w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              disabled={!file}
            >
              解析文件
            </button>
          </div>

          {status && (
            <div className={`p-4 rounded-lg ${status.includes('Error') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
              {status}
            </div>
          )}
        </div>

        <div className="col-span-9 border rounded-lg bg-white shadow-sm">
          {parsedContent ? (
            <div className="p-4">
              <h3 className="text-xl font-semibold mb-4">Parsing Results</h3>
              <div className="mb-4 p-3 border rounded bg-gray-100">
                <h4 className="font-medium mb-2">Document Information</h4>
                <div className="text-sm text-gray-600">
                  <p>Total Pages: {parsedContent.metadata?.total_pages}</p>
                  <p>Parsing Method: {parsedContent.metadata?.parsing_method}</p>
                  <p>Timestamp: {parsedContent.metadata?.timestamp && new Date(parsedContent.metadata.timestamp).toLocaleString()}</p>
                </div>
              </div>

              {blocks.length > 0 && (
                <div className="mb-4 p-3 border rounded bg-blue-50">
                  <h4 className="font-medium mb-2 text-blue-700">Structured Diagnostics</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm text-blue-800">
                    <p>Titles: {titleCount}</p>
                    <p>Tables: {diagnostics.table_count ?? 0}</p>
                    <p>Reference Start Page: {diagnostics.reference_start_page ?? 'N/A'}</p>
                    <p>Repeated Header/Footer: {diagnostics.header_footer_candidates?.length ?? 0}</p>
                  </div>
                </div>
              )}

              <div className="space-y-3 max-h-[calc(100vh-300px)] overflow-y-auto">
                {blocks.length > 0
                  ? blocks.map(renderStructuredBlock)
                  : legacyContent.map(renderLegacyItem)}
              </div>
            </div>
          ) : (
            <RandomImage message="Upload and parse a file to see the results here" />
          )}
        </div>
      </div>
    </div>
  );
};

export default ParseFile;
