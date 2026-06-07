import React, { useEffect, useMemo, useState } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const TEXT_CHUNK_OPTIONS = [
  { value: 'by_pages', label: '按页分块' },
  { value: 'fixed_size', label: '固定长度分块' },
  { value: 'by_paragraphs', label: '按段落分块' },
  { value: 'by_sentences', label: '按句子分块' },
];

const STRUCTURED_CHUNK_OPTIONS = [
  { value: 'by_blocks', label: '按解析块分块' },
  { value: 'by_sections', label: '按章节分块' },
];

const formatTimestamp = (value) => {
  if (!value) {
    return 'N/A';
  }

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

const ChunkFile = () => {
  const [loadedDocuments, setLoadedDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState('');
  const [chunkSource, setChunkSource] = useState('loaded_text');
  const [chunkingOption, setChunkingOption] = useState('by_pages');
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunks, setChunks] = useState(null);
  const [status, setStatus] = useState('');
  const [activeTab, setActiveTab] = useState('chunks');
  const [processingStatus, setProcessingStatus] = useState('');
  const [chunkedDocuments, setChunkedDocuments] = useState([]);

  const availableChunkOptions = useMemo(() => {
    return chunkSource === 'structured_blocks'
      ? STRUCTURED_CHUNK_OPTIONS
      : TEXT_CHUNK_OPTIONS;
  }, [chunkSource]);

  const shouldShowChunkSize = chunkingOption === 'fixed_size' || chunkingOption === 'by_sections';

  useEffect(() => {
    fetchLoadedDocuments();
  }, []);

  useEffect(() => {
    const defaultOption =
      chunkSource === 'structured_blocks'
        ? STRUCTURED_CHUNK_OPTIONS[0].value
        : TEXT_CHUNK_OPTIONS[0].value;
    setChunkingOption(defaultOption);
  }, [chunkSource]);

  const fetchLoadedDocuments = async () => {
    try {
      const [loadedResponse, chunkedResponse] = await Promise.all([
        fetch(`${apiBaseUrl}/documents?type=loaded`),
        fetch(`${apiBaseUrl}/documents?type=chunked`),
      ]);

      if (!loadedResponse.ok || !chunkedResponse.ok) {
        throw new Error('Failed to fetch document lists');
      }

      const loadedData = await loadedResponse.json();
      const chunkedData = await chunkedResponse.json();

      setLoadedDocuments(loadedData.documents || []);

      const chunkedDocsWithDetails = await Promise.all(
        (chunkedData.documents || []).map(async (doc) => {
          try {
            const detailResponse = await fetch(
              `${apiBaseUrl}/documents/${doc.name}?type=chunked`
            );
            if (!detailResponse.ok) {
              return doc;
            }

            const detailData = await detailResponse.json();
            return {
              ...doc,
              total_pages: detailData.total_pages,
              total_chunks: detailData.total_chunks,
              chunking_method: detailData.chunking_method,
              chunk_source: detailData.chunk_source,
              loading_method: detailData.loading_method,
              timestamp: detailData.timestamp,
            };
          } catch (error) {
            console.error(`Error processing document ${doc.name}:`, error);
            return doc;
          }
        })
      );

      setChunkedDocuments(chunkedDocsWithDetails);
    } catch (error) {
      console.error('Error fetching documents:', error);
      setProcessingStatus(`Error fetching documents: ${error.message}`);
    }
  };

  const handleChunk = async () => {
    if (!selectedDoc || !chunkingOption) {
      setStatus('Please select a document and chunking option.');
      return;
    }

    setStatus('Processing...');
    setChunks(null);

    try {
      const docId = selectedDoc.endsWith('.json') ? selectedDoc : `${selectedDoc}.json`;

      const response = await fetch(`${apiBaseUrl}/chunk`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          doc_id: docId,
          chunking_option: chunkingOption,
          chunk_size: chunkSize,
          chunk_source: chunkSource,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `HTTP error! status: ${response.status}`);
      }

      setChunks(data);
      setStatus('Chunking completed successfully.');
      fetchLoadedDocuments();
    } catch (error) {
      console.error('Error:', error);
      setStatus(`Error: ${error.message}`);
    }
  };

  const handleDeleteDocument = async (docName) => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${docName}?type=chunked`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setProcessingStatus('Document deleted successfully.');
      fetchLoadedDocuments();
      if (selectedDoc === docName) {
        setSelectedDoc('');
        setChunks(null);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      setProcessingStatus(`Error deleting document: ${error.message}`);
    }
  };

  const handleViewDocument = async (docName) => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents/${docName}?type=chunked`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `HTTP error! status: ${response.status}`);
      }

      setChunks(data);
      setActiveTab('chunks');
    } catch (error) {
      console.error('Error viewing document:', error);
      setProcessingStatus(`Error viewing document: ${error.message}`);
    }
  };

  const renderChunkCard = (chunk) => {
    const metadata = chunk.metadata || {};
    return (
      <div
        key={metadata.chunk_id || `${metadata.page_number}-${metadata.word_count}`}
        className="p-3 border rounded bg-gray-50"
      >
        <div className="font-medium text-sm text-gray-500 mb-1">
          Chunk {metadata.chunk_id || 'N/A'}
        </div>
        <div className="text-xs text-gray-400 mb-2">
          Page(s): {metadata.page_range || 'N/A'} | Words: {metadata.word_count || 0}
        </div>
        {metadata.block_type && (
          <div className="text-xs text-blue-600 mb-2">
            Block Type: {metadata.block_type}
          </div>
        )}
        {metadata.section_title && (
          <div className="text-xs text-gray-500 mb-2">
            Section: {metadata.section_title}
          </div>
        )}
        <div className="text-sm text-gray-700 whitespace-pre-wrap">{chunk.content}</div>
      </div>
    );
  };

  const renderRightPanel = () => (
    <div className="p-4 w-full h-full flex flex-col">
      <div className="flex mb-4 border-b">
        <button
          className={`px-4 py-2 ${
            activeTab === 'chunks'
              ? 'border-b-2 border-blue-500 text-blue-600'
              : 'text-gray-600'
          }`}
          onClick={() => setActiveTab('chunks')}
        >
          分块预览
        </button>
        <button
          className={`px-4 py-2 ml-4 ${
            activeTab === 'documents'
              ? 'border-b-2 border-blue-500 text-blue-600'
              : 'text-gray-600'
          }`}
          onClick={() => setActiveTab('documents')}
        >
          分块管理
        </button>
      </div>

      {activeTab === 'chunks' ? (
        chunks ? (
          <div className="w-full">
            <div className="mb-4 p-3 border rounded bg-gray-100">
              <h4 className="font-medium mb-2">Document Information</h4>
              <div className="text-sm text-gray-600 space-y-1">
                <p>Filename: {chunks.filename}</p>
                <p>Total Pages: {chunks.total_pages}</p>
                <p>Total Chunks: {chunks.total_chunks}</p>
                <p>Loading Method: {chunks.loading_method}</p>
                <p>Chunk Source: {chunks.chunk_source || 'loaded_text'}</p>
                <p>Chunking Method: {chunks.chunking_method}</p>
                <p>Timestamp: {formatTimestamp(chunks.timestamp)}</p>
              </div>
            </div>
            <div className="space-y-3 max-h-[calc(100vh-300px)] overflow-y-auto">
              {Array.isArray(chunks.chunks) && chunks.chunks.map(renderChunkCard)}
            </div>
          </div>
        ) : (
          <RandomImage message="选择文档并生成分块后，可在这里查看结果" />
        )
      ) : (
        <div className="flex flex-col w-full h-full">
          <h3 className="text-xl font-semibold mb-4">Document Management</h3>
          <div className="space-y-4 w-full">
            {chunkedDocuments.length > 0 ? (
              chunkedDocuments.map((doc) => (
                <div key={doc.name} className="p-4 border rounded-lg bg-gray-50 w-full">
                  <div className="flex justify-between items-start w-full">
                    <div className="flex-grow">
                      <h4 className="font-medium text-lg">{doc.name}</h4>
                      <div className="text-sm text-gray-600 mt-1 space-y-1">
                        <p>Pages: {doc.total_pages || 'N/A'}</p>
                        <p>Chunks: {doc.total_chunks || 'N/A'}</p>
                        <p>Chunk Source: {doc.chunk_source || 'loaded_text'}</p>
                        <p>Chunking Method: {doc.chunking_method || 'N/A'}</p>
                        <p>Loading Method: {doc.loading_method || 'N/A'}</p>
                        <p>Processing Date: {formatTimestamp(doc.timestamp)}</p>
                      </div>
                    </div>
                    <div className="flex space-x-2 ml-4">
                      <button
                        onClick={() => handleViewDocument(doc.name)}
                        className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                      >
                        View
                      </button>
                      <button
                        onClick={() => handleDeleteDocument(doc.name)}
                        className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center text-gray-500 py-8 w-full">
                No chunked documents available.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="p-6">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6"> 检索增强生成工具 </h1>
      <hr />
      <h2 className="text-2xl font-bold mb-6">知识分块</h2>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">选择文档</label>
              <select
                value={selectedDoc}
                onChange={(e) => setSelectedDoc(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="">Choose a document...</option>
                {loadedDocuments.map((doc) => (
                  <option key={doc.name} value={doc.name}>
                    {doc.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">分块来源</label>
              <select
                value={chunkSource}
                onChange={(e) => setChunkSource(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                <option value="loaded_text">已加载文本</option>
                <option value="structured_blocks">结构化解析块</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">分块方法</label>
              <select
                value={chunkingOption}
                onChange={(e) => setChunkingOption(e.target.value)}
                className="block w-full p-2 border rounded"
              >
                {availableChunkOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            {shouldShowChunkSize && (
              <div className="mb-4">
                <label className="block text-sm font-medium mb-1">Chunk Size</label>
                <input
                  type="number"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  className="block w-full p-2 border rounded"
                  min="100"
                  max="5000"
                />
              </div>
            )}

            <button
              onClick={handleChunk}
              className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300"
              disabled={!selectedDoc}
            >
              产生分块
            </button>
          </div>

          {status && (
            <div
              className={`p-4 rounded-lg ${
                status.includes('Error')
                  ? 'bg-red-100 text-red-700'
                  : 'bg-green-100 text-green-700'
              }`}
            >
              {status}
            </div>
          )}

          {processingStatus && (
            <div className="p-4 rounded-lg bg-gray-100 text-gray-700">
              {processingStatus}
            </div>
          )}
        </div>

        <div className="col-span-9 border rounded-lg bg-white shadow-sm">
          {renderRightPanel()}
        </div>
      </div>
    </div>
  );
};

export default ChunkFile;
