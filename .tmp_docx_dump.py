from zipfile import ZipFile
from lxml import etree
p=r'E:\project\数据挖掘第二组报告\第二组数据挖掘与信息检索实验报告.docx'
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(p) as z:
    xml=z.read('word/document.xml')
root=etree.fromstring(xml)
for para in root.xpath('//w:p', namespaces=ns):
    txt=''.join(para.xpath('.//w:t/text()', namespaces=ns)).strip()
    if txt:
        print(txt)
