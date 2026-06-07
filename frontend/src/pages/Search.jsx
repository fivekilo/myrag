import React, { useEffect, useState } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const Search = () => {
  const [query, setQuery] = useState('');
  const [collection, setCollection] = useState('');
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [topK, setTopK] = useState(3);
  const [threshold, setThreshold] = useState(0.7);
  const [collections, setCollections] = useState([]);
  const [providers, setProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('chroma');
  const [wordCountThreshold, setWordCountThreshold] = useState(100);
  const [saveResults, setSaveResults] = useState(false);
  const [status, setStatus] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const providersResponse = await fetch(`${apiBaseUrl}/providers`);
        if (!providersResponse.ok) {
          throw new Error(`Failed to load providers: ${providersResponse.status}`);
        }
        const providersData = await providersResponse.json();
        const nextProviders = Array.isArray(providersData.providers) ? providersData.providers : [];
        setProviders(nextProviders);

        if (nextProviders.length > 0 && !nextProviders.some((provider) => provider.id === selectedProvider)) {
          setSelectedProvider(nextProviders[0].id);
          return;
        }

        const collectionsResponse = await fetch(`${apiBaseUrl}/collections?provider=${selectedProvider}`);
        if (!collectionsResponse.ok) {
          throw new Error(`Failed to load collections: ${collectionsResponse.status}`);
        }
        const collectionsData = await collectionsResponse.json();
        setCollections(Array.isArray(collectionsData.collections) ? collectionsData.collections : []);
      } catch (error) {
        console.error('Error fetching data:', error);
        setProviders([]);
        setCollections([]);
        setStatus(`Error loading search data: ${error.message}`);
      }
    };

    fetchData();
  }, [selectedProvider]);

  const handleSearch = async () => {
    if (!query || !collection) {
      setStatus('Please choose a collection and enter a search query.');
      return;
    }

    setIsSearching(true);
    setStatus('');

    try {
      const response = await fetch(`${apiBaseUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          collection_id: collection,
          top_k: topK,
          threshold,
          word_count_threshold: wordCountThreshold,
          save_results: saveResults
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `HTTP error! status: ${response.status}`);
      }

      const nextResults = Array.isArray(data.results?.results)
        ? data.results.results
        : Array.isArray(data.results)
          ? data.results
          : [];

      setResults(nextResults);

      if (nextResults.length === 0) {
        setStatus('No matching results were found.');
      } else if (saveResults && data.saved_filepath) {
        setStatus(`Search completed. Results saved to ${data.saved_filepath}`);
      } else {
        setStatus('Search completed.');
      }
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
      setStatus(`Search failed: ${error.message}`);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSaveResults = async () => {
    if (results.length === 0) {
      setStatus('There are no search results to save.');
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/save-search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          collection_id: collection,
          results
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `HTTP error! status: ${response.status}`);
      }

      setStatus(`Results saved to: ${data.saved_filepath}`);
    } catch (error) {
      console.error('Save error:', error);
      setStatus(`Saving failed: ${error.message}`);
    }
  };

  return (
    <div className="p-6 text-gray-900">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6">检索增强生成工具</h1>
      <hr />
      <h2 className="text-2xl font-bold mb-6 text-gray-900">相似性检索</h2>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm text-gray-900">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">用户查询</label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your search query..."
                  className="block w-full p-2 border rounded h-32 resize-none text-gray-900 bg-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">向量数据库</label>
                <select
                  value={selectedProvider}
                  onChange={(e) => setSelectedProvider(e.target.value)}
                  className="block w-full p-2 border rounded text-gray-900 bg-white"
                >
                  {providers.map((provider) => (
                    <option key={provider.id} value={provider.id}>
                      {provider.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">集合</label>
                <select
                  value={collection}
                  onChange={(e) => setCollection(e.target.value)}
                  className="block w-full p-2 border rounded text-gray-900 bg-white"
                >
                  <option value="">Choose a collection...</option>
                  {collections.map((coll) => (
                    <option key={coll.id} value={coll.id}>
                      {coll.name} ({coll.count} documents)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Top K Results</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value, 10) || 1)}
                  min="1"
                  max="10"
                  className="block w-full p-2 border rounded text-gray-900 bg-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Similarity Threshold: {threshold}</label>
                <input
                  type="range"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  min="0"
                  max="1"
                  step="0.1"
                  className="block w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Minimum Word Count: {wordCountThreshold}</label>
                <input
                  type="range"
                  value={wordCountThreshold}
                  onChange={(e) => setWordCountThreshold(parseInt(e.target.value, 10))}
                  min="0"
                  max="500"
                  step="10"
                  className="block w-full"
                />
              </div>

              <div className="mt-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={saveResults}
                    onChange={(e) => setSaveResults(e.target.checked)}
                    className="form-checkbox h-4 w-4 text-blue-600"
                  />
                  <span className="text-sm font-medium">保存搜索结果</span>
                </label>
              </div>

              <button
                onClick={handleSearch}
                disabled={isSearching}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
              >
                {isSearching ? 'Searching...' : '搜索'}
              </button>
            </div>
          </div>

          {status && (
            <div className={`p-4 rounded-lg ${status.toLowerCase().includes('error') || status.toLowerCase().includes('failed') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
              {status}
            </div>
          )}
        </div>

        <div className="col-span-9 border rounded-lg bg-white shadow-sm text-gray-900">
          {results.length > 0 ? (
            <div className="p-4 text-gray-900">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-semibold text-gray-900">Search Results</h3>
                <button
                  onClick={handleSaveResults}
                  className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  保存搜索结果
                </button>
              </div>
              <div className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
                {results.map((result, idx) => (
                  <div key={idx} className="p-4 border rounded bg-gray-50 text-gray-900">
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-sm text-gray-500">
                        Match Score: {typeof result.score === 'number' ? `${(result.score * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                      <div className="text-sm text-gray-500">
                        <div>Source: {result.metadata?.source || result.metadata?.filename || 'Unknown'}</div>
                        <div>Page: {result.metadata?.page || result.metadata?.page_number || 'N/A'}</div>
                        <div>Chunk: {result.metadata?.chunk || result.metadata?.chunk_id || 'N/A'}</div>
                      </div>
                    </div>
                    <p className="text-sm whitespace-pre-wrap text-gray-900">{result.text || result.metadata?.content || ''}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <RandomImage message="Search results will appear here" />
          )}
        </div>
      </div>
    </div>
  );
};

export default Search;
