import React, { useEffect, useState } from 'react';
import RandomImage from '../components/RandomImage';
import { apiBaseUrl } from '../config/config';

const Indexing = () => {
  const [embeddingFile, setEmbeddingFile] = useState('');
  const [vectorDb, setVectorDb] = useState('chroma');
  const [indexMode, setIndexMode] = useState('standard');
  const [status, setStatus] = useState('');
  const [embeddedFiles, setEmbeddedFiles] = useState([]);
  const [indexingResult, setIndexingResult] = useState(null);
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState('');
  const [providers, setProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('chroma');

  const dbConfigs = {
    pinecone: { modes: ['standard', 'hybrid'] },
    milvus: { modes: ['flat', 'ivf_flat', 'ivf_sq8', 'hnsw'] },
    qdrant: { modes: ['hnsw', 'custom'] },
    weaviate: { modes: ['hnsw', 'flat'] },
    chroma: { modes: ['hnsw', 'standard'] },
    faiss: { modes: ['flat', 'ivf_flat', 'hnsw'] } // 【修改此处：ivf -> ivf_flat】
  };

  useEffect(() => {
    fetchEmbeddedFiles();
  }, []);

  useEffect(() => {
    const nextModes = dbConfigs[vectorDb]?.modes || ['standard'];
    setIndexMode(nextModes[0]);
  }, [vectorDb]);

  useEffect(() => {
    setVectorDb(selectedProvider);
  }, [selectedProvider]);

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
        setStatus(`Error loading indexing data: ${error.message}`);
      }
    };

    fetchData();
  }, [selectedProvider]);

  const fetchEmbeddedFiles = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/list-embedded`);
      if (!response.ok) {
        throw new Error(`Failed to load embedded files: ${response.status}`);
      }
      const data = await response.json();
      const documents = Array.isArray(data.documents) ? data.documents : [];
      setEmbeddedFiles(
        documents.map((doc) => ({
          ...doc,
          id: doc.name,
          displayName: doc.name
        }))
      );
    } catch (error) {
      console.error('Error fetching embedded files:', error);
      setEmbeddedFiles([]);
      setStatus(`Error loading embedding files: ${error.message}`);
    }
  };

  const fetchCollections = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/collections?provider=${selectedProvider}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setCollections(Array.isArray(data.collections) ? data.collections : []);
    } catch (error) {
      console.error('Error fetching collections:', error);
      setCollections([]);
      setStatus(`Error loading collections: ${error.message}`);
    }
  };

  const handleIndex = async () => {
    if (!embeddingFile) {
      setStatus('Please select an embedding file');
      return;
    }

    setStatus('Indexing...');
    try {
      const response = await fetch(`${apiBaseUrl}/index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          fileId: embeddingFile,
          vectorDb: selectedProvider,
          indexMode
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `HTTP error! status: ${response.status}`);
      }

      setIndexingResult(data);
      await fetchCollections();
      if (data.collection_name) {
        setSelectedCollection(data.collection_name);
      }
      setStatus('Indexing completed successfully');
    } catch (error) {
      console.error('Error indexing:', error);
      setIndexingResult(null);
      setStatus(`Error during indexing: ${error.message}`);
    }
  };

  const handleDisplay = async (collectionName) => {
    if (!collectionName) {
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/collections/${selectedProvider}/${collectionName}`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `HTTP error! status: ${response.status}`);
      }

      setIndexingResult({
        database: selectedProvider,
        collection_name: data.name,
        total_vectors: data.num_entities,
        index_size: data.num_entities,
        processing_time: data.processing_time
      });
      setStatus('Collection loaded successfully');
    } catch (error) {
      console.error('Error displaying collection:', error);
      setStatus(`Error displaying collection: ${error.message}`);
    }
  };

  const handleDelete = async (collectionName) => {
    if (!collectionName) {
      return;
    }

    if (!window.confirm(`Are you sure you want to delete collection "${collectionName}"?`)) {
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/collections/${selectedProvider}/${collectionName}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || `HTTP error! status: ${response.status}`);
      }

      setSelectedCollection('');
      setIndexingResult(null);
      await fetchCollections();
      setStatus('Collection deleted successfully');
    } catch (error) {
      console.error('Error deleting collection:', error);
      setStatus(`Error deleting collection: ${error.message}`);
    }
  };

  return (
    <div className="p-6 text-gray-900">
      <h1 className="text-blue-500 text-3xl font-bold text-center mb-6">检索增强生成工具</h1>
      <hr />
      <h2 className="text-2xl font-bold mb-6 text-gray-900">向量库索引</h2>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-3">
          <div className="p-4 border rounded-lg bg-white shadow-sm space-y-4 text-gray-900">
            <div>
              <label className="block text-sm font-medium mb-1">待索引的嵌入文件</label>
              <select
                value={embeddingFile}
                onChange={(e) => setEmbeddingFile(e.target.value)}
                className="block w-full p-2 border rounded text-gray-900 bg-white"
              >
                <option value="">Choose a file...</option>
                {embeddedFiles.map((file) => (
                  <option key={file.name} value={file.name}>
                    {file.displayName}
                  </option>
                ))}
              </select>
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
              <label className="block text-sm font-medium mb-1">索引模式</label>
              <select
                value={indexMode}
                onChange={(e) => setIndexMode(e.target.value)}
                className="block w-full p-2 border rounded text-gray-900 bg-white"
              >
                {(dbConfigs[vectorDb]?.modes || ['standard']).map((mode) => (
                  <option key={mode} value={mode}>
                    {mode.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <button
                onClick={handleIndex}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
                disabled={!embeddingFile}
              >
                执行索引
              </button>

              <div>
                <label className="block text-sm font-medium mb-1">索引集合</label>
                <select
                  value={selectedCollection}
                  onChange={(e) => setSelectedCollection(e.target.value)}
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

              <button
                onClick={() => handleDisplay(selectedCollection)}
                disabled={!selectedCollection}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
              >
                查看集合
              </button>

              <button
                onClick={() => handleDelete(selectedCollection)}
                disabled={!selectedCollection}
                className="w-full px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-red-300"
              >
                删除集合
              </button>
            </div>

            {status && (
              <div className="mt-4 p-3 rounded border bg-gray-50 text-gray-900">
                <p className="text-sm text-gray-900">{status}</p>
              </div>
            )}
          </div>
        </div>

        <div className="col-span-9 border rounded-lg bg-white shadow-sm text-gray-900">
          {indexingResult ? (
            <div className="p-4 text-gray-900">
              <h3 className="text-xl font-semibold mb-4 text-gray-900">索引结果</h3>
              <div className="space-y-3">
                <div className="p-3 border rounded bg-gray-50">
                  <div className="text-sm text-gray-700">
                    <p>Database: {indexingResult.database}</p>
                    {indexingResult.index_mode && <p>Index Mode: {indexingResult.index_mode}</p>}
                    <p>Total Vectors: {indexingResult.total_vectors}</p>
                    <p>Index Size: {indexingResult.index_size}</p>
                    {indexingResult.processing_time && <p>Processing Time: {indexingResult.processing_time}s</p>}
                    <p>Collection Name: {indexingResult.collection_name}</p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <RandomImage message="Indexing results will appear here" />
          )}
        </div>
      </div>
    </div>
  );
};

export default Indexing;
