import { useState, useEffect } from 'react';
import { apiBaseUrl } from '../config/config';

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    loadedDocs: 0,
    chunkedDocs: 0,
    embeddedDocs: 0,
    collections: 0,
    searchResults: 0,
  });
  const [activeTab, setActiveTab] = useState('loaded');
  const [docLists, setDocLists] = useState({
    loaded: [],
    chunked: [],
    embedded: [],
    collections: [],
    searchResults: [],
  });
  const [searchKeyword, setSearchKeyword] = useState('');
  const [selectedItems, setSelectedItems] = useState([]);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    fetchAllStats();
  }, []);

  const fetchAllStats = async () => {
    setLoading(true);
    try {
      const [loadedRes, chunkedRes, embeddedRes, collectionsRes, searchRes] = await Promise.all([
        fetch(`${apiBaseUrl}/documents?type=loaded`).then(r => r.json()).catch(() => ({ documents: [] })),
        fetch(`${apiBaseUrl}/documents?type=chunked`).then(r => r.json()).catch(() => ({ documents: [] })),
        fetch(`${apiBaseUrl}/list-embedded`).then(r => r.json()).catch(() => ({ documents: [] })),
        fetch(`${apiBaseUrl}/collections`).then(r => r.json()).catch(() => ({ collections: [] })),
        fetch(`${apiBaseUrl}/search-results`).then(r => r.json()).catch(() => ({ files: [] })),
      ]);

      setStats({
        loadedDocs: loadedRes.documents?.length || 0,
        chunkedDocs: chunkedRes.documents?.length || 0,
        embeddedDocs: embeddedRes.documents?.length || 0,
        collections: collectionsRes.collections?.length || 0,
        searchResults: searchRes.files?.length || 0,
      });

      const loadedDocs = (loadedRes.documents || []).map(doc => ({
        name: doc.name || doc.id,
        method: doc.metadata?.loading_method || '-',
        pages: doc.metadata?.total_pages || '-',
        chunks: doc.metadata?.total_chunks || '-',
        time: doc.metadata?.timestamp || '-',
      }));
      loadedDocs.sort((a, b) => (b.time > a.time ? 1 : -1));

      const chunkedDocs = (chunkedRes.documents || []).map(doc => ({
        name: doc.name || doc.id,
        type: doc.type || 'chunked',
      }));

      const embeddedDocs = (embeddedRes.documents || []).map(doc => ({
        name: doc.name,
        model: doc.metadata?.embedding_model || '-',
        provider: doc.metadata?.embedding_provider || '-',
        dimension: doc.metadata?.vector_dimension || '-',
        time: doc.metadata?.embedding_timestamp || '-',
      }));

      const collectionsList = (collectionsRes.collections || []).map(col => ({
        name: typeof col === 'string' ? col : col.name || col.id || '-',
      }));

      const searchResultsList = (searchRes.files || []).map(file => ({
        name: file.name || file.id,
        time: file.timestamp || '-',
      }));
      searchResultsList.sort((a, b) => (b.time > a.time ? 1 : -1));

      setDocLists({
        loaded: loadedDocs,
        chunked: chunkedDocs,
        embedded: embeddedDocs,
        collections: collectionsList,
        searchResults: searchResultsList,
      });
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
    } finally {
      setLoading(false);
    }
  };

  // 切换 tab 时重置搜索和选择状态
  const handleTabChange = (key) => {
    setActiveTab(key);
    setSearchKeyword('');
    setSelectedItems([]);
  };

  // 当前 tab 是否支持删除
  const canDelete = activeTab !== 'searchResults';

  // 获取过滤后的列表
  const getFilteredList = () => {
    const list = docLists[activeTab] || [];
    if (!searchKeyword) return list;
    return list.filter(item =>
      item.name.toLowerCase().includes(searchKeyword.toLowerCase())
    );
  };

  // 选择/取消选择单项
  const toggleSelect = (name) => {
    setSelectedItems(prev =>
      prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    );
  };

  // 全选/取消全选
  const toggleSelectAll = (filteredList) => {
    if (selectedItems.length === filteredList.length) {
      setSelectedItems([]);
    } else {
      setSelectedItems(filteredList.map(item => item.name));
    }
  };

  // 执行删除
  const handleDelete = async (items) => {
    setDeleting(true);
    try {
      for (const item of items) {
        if (activeTab === 'loaded') {
          await fetch(`${apiBaseUrl}/documents/${encodeURIComponent(item.name)}?type=loaded`, { method: 'DELETE' });
        } else if (activeTab === 'chunked') {
          await fetch(`${apiBaseUrl}/documents/${encodeURIComponent(item.name)}?type=chunked`, { method: 'DELETE' });
        } else if (activeTab === 'embedded') {
          await fetch(`${apiBaseUrl}/embedded-docs/${encodeURIComponent(item.name)}`, { method: 'DELETE' });
        } else if (activeTab === 'collections') {
          await fetch(`${apiBaseUrl}/collections/chroma/${encodeURIComponent(item.name)}`, { method: 'DELETE' });
        }
      }
      setSelectedItems([]);
      setDeleteConfirm(null);
      await fetchAllStats();
    } catch (error) {
      console.error('Error deleting:', error);
    } finally {
      setDeleting(false);
    }
  };

  const tabs = [
    { key: 'loaded', title: '已导入文档', value: stats.loadedDocs, color: 'bg-blue-500', borderColor: 'border-blue-500' },
    { key: 'chunked', title: '已分块文档', value: stats.chunkedDocs, color: 'bg-green-500', borderColor: 'border-green-500' },
    { key: 'embedded', title: '已嵌入文档', value: stats.embeddedDocs, color: 'bg-purple-500', borderColor: 'border-purple-500' },
    { key: 'collections', title: '向量库集合', value: stats.collections, color: 'bg-orange-500', borderColor: 'border-orange-500' },
    { key: 'searchResults', title: '检索记录', value: stats.searchResults, color: 'bg-red-500', borderColor: 'border-red-500' },
  ];

  // 渲染文档列表
  const renderDocList = () => {
    const filteredList = getFilteredList();
    if (!filteredList || filteredList.length === 0) {
      return <p className="text-gray-400 text-center py-8">暂无数据</p>;
    }

    const headers = {
      loaded: ['文件名', '加载方式', '页数', '分块数', '时间'],
      chunked: ['文件名'],
      embedded: ['文件名', '嵌入模型', '提供者', '维度', '时间'],
      collections: ['集合名称'],
      searchResults: ['检索记录', '时间'],
    };

    const renderRow = (item, index) => {
      const isSelected = selectedItems.includes(item.name);
      const cells = [];

      if (activeTab === 'loaded') {
        cells.push(
          <td key="name" className="px-4 py-3 font-medium text-gray-800 truncate max-w-xs">{item.name}</td>,
          <td key="method" className="px-4 py-3 text-gray-600">{item.method}</td>,
          <td key="pages" className="px-4 py-3 text-gray-600">{item.pages}</td>,
          <td key="chunks" className="px-4 py-3 text-gray-600">{item.chunks}</td>,
          <td key="time" className="px-4 py-3 text-gray-500 text-xs">{item.time !== '-' ? new Date(item.time).toLocaleString('zh-CN') : '-'}</td>
        );
      } else if (activeTab === 'chunked') {
        cells.push(<td key="name" className="px-4 py-3 font-medium text-gray-800 truncate max-w-md">{item.name}</td>);
      } else if (activeTab === 'embedded') {
        cells.push(
          <td key="name" className="px-4 py-3 font-medium text-gray-800 truncate max-w-xs">{item.name}</td>,
          <td key="model" className="px-4 py-3 text-gray-600">{item.model}</td>,
          <td key="provider" className="px-4 py-3 text-gray-600">{item.provider}</td>,
          <td key="dimension" className="px-4 py-3 text-gray-600">{item.dimension}</td>,
          <td key="time" className="px-4 py-3 text-gray-500 text-xs">{item.time !== '-' ? new Date(item.time).toLocaleString('zh-CN') : '-'}</td>
        );
      } else if (activeTab === 'collections') {
        cells.push(<td key="name" className="px-4 py-3 font-medium text-gray-800">{item.name}</td>);
      } else if (activeTab === 'searchResults') {
        cells.push(
          <td key="name" className="px-4 py-3 font-medium text-gray-800 truncate max-w-md">{item.name}</td>,
          <td key="time" className="px-4 py-3 text-gray-500 text-xs">{item.time !== '-' ? new Date(item.time).toLocaleString('zh-CN') : '-'}</td>
        );
      }

      return (
        <tr key={index} className={`border-b hover:bg-gray-50 ${isSelected ? 'bg-blue-50' : ''}`}>
          {canDelete && (
            <td className="px-4 py-3">
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => toggleSelect(item.name)}
                className="w-4 h-4 rounded border-gray-300"
              />
            </td>
          )}
          <td className="px-4 py-3 text-gray-400">{index + 1}</td>
          {cells}
          {canDelete && (
            <td className="px-4 py-3">
              <button
                onClick={() => setDeleteConfirm({ type: 'single', items: [item] })}
                className="text-red-500 hover:text-red-700 text-xs font-medium"
              >
                删除
              </button>
            </td>
          )}
        </tr>
      );
    };

    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-gray-500 uppercase bg-gray-50">
            <tr>
              {canDelete && (
                <th className="px-4 py-3 w-10">
                  <input
                    type="checkbox"
                    checked={selectedItems.length === filteredList.length && filteredList.length > 0}
                    onChange={() => toggleSelectAll(filteredList)}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                </th>
              )}
              <th className="px-4 py-3 w-12">#</th>
              {headers[activeTab].map(h => (
                <th key={h} className="px-4 py-3">{h}</th>
              ))}
              {canDelete && <th className="px-4 py-3 w-16">操作</th>}
            </tr>
          </thead>
          <tbody>
            {filteredList.map((item, index) => renderRow(item, index))}
          </tbody>
        </table>
      </div>
    );
  };

  // 确认删除对话框
  const renderDeleteConfirm = () => {
    if (!deleteConfirm) return null;
    const count = deleteConfirm.items.length;
    const isSingle = deleteConfirm.type === 'single';

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">确认删除</h3>
          <p className="text-gray-600 mb-4">
            {isSingle
              ? `确定要删除「${deleteConfirm.items[0].name}」吗？`
              : `确定要删除选中的 ${count} 个项目吗？`
            }
          </p>
          <p className="text-sm text-red-500 mb-4">此操作不可逆，删除后无法恢复。</p>
          <div className="flex justify-end gap-3">
            <button
              onClick={() => setDeleteConfirm(null)}
              disabled={deleting}
              className="px-4 py-2 text-sm text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200"
            >
              取消
            </button>
            <button
              onClick={() => handleDelete(deleteConfirm.items)}
              disabled={deleting}
              className="px-4 py-2 text-sm text-white bg-red-500 rounded-md hover:bg-red-600 disabled:opacity-50"
            >
              {deleting ? '删除中...' : '确认删除'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">数据概览</h1>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        {tabs.map((tab) => (
          <div
            key={tab.key}
            onClick={() => handleTabChange(tab.key)}
            className={`bg-white rounded-lg p-5 cursor-pointer transition-all duration-200 border-2 ${
              activeTab === tab.key
                ? `${tab.borderColor} shadow-lg`
                : 'border-transparent shadow-md hover:shadow-lg'
            }`}
          >
            <div className={`w-10 h-10 rounded-full ${tab.color} flex items-center justify-center mb-3`}>
              <span className="text-white font-bold text-sm">
                {loading ? '...' : tab.value}
              </span>
            </div>
            <h3 className="text-gray-600 text-sm font-medium">{tab.title}</h3>
            <p className="text-2xl font-bold text-gray-800 mt-1">
              {loading ? (
                <span className="inline-block w-12 h-7 bg-gray-200 rounded animate-pulse"></span>
              ) : (
                tab.value
              )}
            </p>
          </div>
        ))}
      </div>

      {/* 文档列表区域 */}
      <div className="bg-white rounded-lg shadow-md p-5">
        {/* 标题栏 + 搜索 + 批量操作 */}
        <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
          <h2 className="text-lg font-semibold text-gray-700">
            {tabs.find(t => t.key === activeTab)?.title}列表
            <span className="text-sm font-normal text-gray-400 ml-2">
              （共 {tabs.find(t => t.key === activeTab)?.value} 条）
            </span>
          </h2>
          <div className="flex items-center gap-2">
            {canDelete && selectedItems.length > 0 && (
              <button
                onClick={() => {
                  const items = getFilteredList().filter(item => selectedItems.includes(item.name));
                  setDeleteConfirm({ type: 'batch', items });
                }}
                className="px-3 py-1.5 text-sm text-white bg-red-500 rounded-md hover:bg-red-600"
              >
                批量删除 ({selectedItems.length})
              </button>
            )}
            <input
              type="text"
              placeholder="搜索文件名..."
              value={searchKeyword}
              onChange={(e) => setSearchKeyword(e.target.value)}
              className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-400 w-48"
            />
          </div>
        </div>

        {/* 表格 */}
        {loading ? (
          <div className="space-y-3">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-10 bg-gray-100 rounded animate-pulse"></div>
            ))}
          </div>
        ) : renderDocList()}
      </div>

      {/* 删除确认弹窗 */}
      {renderDeleteConfirm()}
    </div>
  );
};

export default Dashboard;
