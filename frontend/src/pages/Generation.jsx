import React, { useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiBaseUrl } from '../config/config';

const MarkdownViewer = ({ markdownText }) => (
  <div className="markdown-container">
    <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdownText || ''}</ReactMarkdown>
  </div>
);

const getModelDisplayName = (id, fallbackName) => {
  if (id === 'deepseek-v4-flash') return 'DeepSeek V4 Flash';
  if (id === 'deepseek-v4-pro') return 'DeepSeek V4 Pro';
  if (id === 'deepseek-chat') return 'DeepSeek Chat';
  if (id === 'deepseek-reasoner') return 'DeepSeek Reasoner';
  if (id === 'deepseek-v3') return 'DeepSeek V3';
  if (id === 'deepseek-r1') return 'DeepSeek R1';
  return fallbackName || id;
};

const Generation = () => {
  const location = useLocation();
  const [provider, setProvider] = useState('');
  const [modelName, setModelName] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [models, setModels] = useState({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [response, setResponse] = useState('');
  const [status, setStatus] = useState('');
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [searchFiles, setSearchFiles] = useState([]);
  const [showReasoning, setShowReasoning] = useState(true);
  const [loadModel, setLoadModel] = useState(false);

  const currentProviderModels = useMemo(() => {
    return Object.entries(models?.[provider] || {});
  }, [models, provider]);

  const hasSearchContext = Array.isArray(searchResults) && searchResults.length > 0;

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [modelsResponse, filesResponse] = await Promise.all([
          fetch(`${apiBaseUrl}/generation/models`),
          fetch(`${apiBaseUrl}/search-results`),
        ]);

        if (!modelsResponse.ok) {
          throw new Error(`加载模型列表失败: ${modelsResponse.status}`);
        }
        if (!filesResponse.ok) {
          throw new Error(`加载检索结果列表失败: ${filesResponse.status}`);
        }

        const modelsData = await modelsResponse.json();
        const filesData = await filesResponse.json();

        setModels(modelsData?.models || {});
        setSearchFiles(Array.isArray(filesData?.files) ? filesData.files : []);
      } catch (error) {
        console.error('Error fetching data:', error);
        setStatus(`获取数据失败: ${error.message}`);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    const loadSearchResults = async () => {
      if (!selectedFile) {
        return;
      }

      try {
        const resultResponse = await fetch(`${apiBaseUrl}/search-results/${selectedFile}`);
        if (!resultResponse.ok) {
          throw new Error(`加载检索结果失败: ${resultResponse.status}`);
        }

        const data = await resultResponse.json();
        setQuery(data?.query || '');
        setSearchResults(Array.isArray(data?.results) ? data.results : []);
      } catch (error) {
        console.error('Error loading search results:', error);
        setStatus(`加载搜索结果失败: ${error.message}`);
      }
    };

    loadSearchResults();
  }, [selectedFile]);

  useEffect(() => {
    if (!location.state) {
      return;
    }

    const { query: searchQuery, results } = location.state;
    if (searchQuery) {
      setQuery(searchQuery);
    }
    if (Array.isArray(results)) {
      setSearchResults(results);
    }
  }, [location]);

  useEffect(() => {
    setModelName('');
  }, [provider]);

  const handleGenerate = async () => {
    if (!provider || !modelName) {
      setStatus('请选择生成模型');
      return;
    }

    if (!query.trim()) {
      setStatus('请输入问题并确保有搜索结果');
      return;
    }

    setIsGenerating(true);
    setStatus('');

    try {
      const generateResponse = await fetch(`${apiBaseUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          provider,
          model_name: modelName,
          search_results: Array.isArray(searchResults) ? searchResults : [],
          load_model: loadModel,
          api_key: apiKey || null,
          show_reasoning: showReasoning,
        }),
      });

      const data = await generateResponse.json();
      if (!generateResponse.ok) {
        throw new Error(data?.detail || `HTTP error! status: ${generateResponse.status}`);
      }

      setResponse(data?.response || '');
      setStatus(
        `生成完成！modelStatus: ${loadModel} 结果已保存至: ${data?.saved_filepath || '未返回路径'}`
      );
    } catch (error) {
      console.error('Generation error:', error);
      setStatus(`生成失败: ${error.message}`);
    } finally {
      setIsGenerating(false);
      setLoadModel(false);
    }
  };

  return (
    <div className="p-6 text-gray-900">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6"> 检索增强生成工具 </h1>
      <hr />
      <h2 className="text-2xl font-bold mb-6 text-gray-900">响应生成</h2>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-4 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm text-gray-900">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">提问</label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your question..."
                  className="block w-full p-2 border rounded h-32 resize-none text-gray-900 bg-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">检索文档（可选）</label>
                <select
                  value={selectedFile}
                  onChange={(e) => setSelectedFile(e.target.value)}
                  className="block w-full p-2 border rounded text-gray-900 bg-white"
                >
                  <option value="">Select search results file...</option>
                  {searchFiles.map((file) => (
                    <option key={file.id} value={file.id}>
                      {file.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">生成模型提供方</label>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="block w-full p-2 border rounded text-gray-900 bg-white"
                >
                  <option value="">Select provider...</option>
                  {Object.keys(models || {}).map((providerKey) => (
                    <option key={providerKey} value={providerKey}>
                      {providerKey}
                    </option>
                  ))}
                </select>
              </div>

              {provider && (
                <div>
                  <label className="block text-sm font-medium mb-1">生成模型</label>
                  <select
                    value={modelName}
                    onChange={(e) => {
                      setModelName(e.target.value);
                      setLoadModel(true);
                    }}
                    className="block w-full p-2 border rounded text-gray-900 bg-white"
                  >
                    <option value="">Select model...</option>
                    {currentProviderModels.map(([id, name]) => (
                      <option key={id} value={id}>
                        {getModelDisplayName(id, name)}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {(provider === 'openai' || provider === 'deepseek') && (
                <div>
                  <label className="block text-sm font-medium mb-1">API Key</label>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="Enter your API key..."
                    className="block w-full p-2 border rounded text-gray-900 bg-white"
                  />
                </div>
              )}

              {provider === 'deepseek' &&
                modelName &&
                (modelName === 'deepseek-r1' || modelName === 'deepseek-reasoner') && (
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="showReasoning"
                      checked={showReasoning}
                      onChange={(e) => setShowReasoning(e.target.checked)}
                      className="rounded border-gray-300 text-green-500 focus:ring-green-500"
                    />
                    <label htmlFor="showReasoning" className="text-sm font-medium">
                      显示思维链过程
                    </label>
                  </div>
                )}

              <button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-green-300"
              >
                {isGenerating ? '生成回答中...' : '生成回答'}
              </button>

              {status && (
                <div
                  className={`p-4 rounded-lg ${
                    status.includes('失败') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                  }`}
                >
                  {status}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="col-span-8">
          {hasSearchContext ? (
            <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm text-gray-900">
              <h3 className="text-xl font-semibold mb-4 text-gray-900">检索的上下文</h3>
              <div className="space-y-4 max-h-[300px] overflow-y-auto">
                {searchResults.map((result, idx) => {
                  const metadata = result?.metadata || {};
                  const score = typeof result?.score === 'number' ? result.score : 0;
                  return (
                    <div key={`${idx}-${metadata.page || 'na'}`} className="p-4 border rounded bg-gray-50 text-gray-900">
                      <div className="flex justify-between items-start mb-2">
                        <span className="font-medium text-sm text-gray-500">
                          Match Score: {(score * 100).toFixed(1)}%
                        </span>
                        <div className="text-sm text-gray-500">
                          <div>Source: {metadata.source || 'N/A'}</div>
                          <div>Page: {metadata.page || metadata.page_number || 'N/A'}</div>
                        </div>
                      </div>
                      <p className="text-sm whitespace-pre-wrap text-gray-900">{result?.text || ''}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm text-gray-900">
              <h3 className="text-xl font-semibold mb-4 text-gray-900">无检索上下文</h3>
            </div>
          )}

          {response && (
            <div className="p-4 border rounded-lg bg-white shadow-sm text-gray-900">
              <h3 className="text-xl font-semibold mb-4 text-gray-900">生成的回答</h3>
              <div className="p-4 border rounded bg-gray-50 text-gray-900">
                <div className="whitespace-pre-wrap text-gray-900">
                  <MarkdownViewer markdownText={response} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Generation;
