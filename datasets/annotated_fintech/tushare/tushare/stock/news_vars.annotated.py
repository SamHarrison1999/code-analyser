# -*- coding:utf-8 -*- 
# 🧠 ML Signal: URL pattern with placeholders, useful for learning URL structures

LATEST_URL = '%sroll.news.%s/interface/%s?col=43&spec=&type=&ch=03&k=&offset_page=0&offset_num=0&num=%s&asc=&page=1&r=0.%s'
# 🧠 ML Signal: Column names for data extraction, useful for learning data schemas
LATEST_COLS = ['classify','title','time','url']
LATEST_COLS_C = ['classify','title','time','url','content']
# 🧠 ML Signal: Extended column names, useful for learning data schemas
NOTICE_INFO_URL = '%s%s/corp/view/%s?stock_str=%s'
# 🧠 ML Signal: URL pattern with placeholders, useful for learning URL structures
# 🧠 ML Signal: Column names for data extraction, useful for learning data schemas
NOTICE_INFO_CLS = ['title', 'type', 'date', 'url']
GUBA_SINA_URL = '%sguba.%s'
GUBA_SINA_COLS = ['title', 'content', 'ptime', 'rcounts']