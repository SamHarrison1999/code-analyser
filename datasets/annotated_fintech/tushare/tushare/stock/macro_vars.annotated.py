# -*- coding:utf-8 -*- 
# 🧠 ML Signal: Usage of dictionary to map protocol types to their URL prefixes

P_TYPE = {'http':'http://','ftp':'ftp://'}
# 🧠 ML Signal: Usage of dictionary to map domain keys to their full domain names
DOMAINS = {'sina':'sina.com.cn','sinahq':'sinajs.cn','ifeng':'ifeng.com'}
MACRO_TYPE = ['nation','price','fininfo']
# 🧠 ML Signal: Usage of list to store macroeconomic data categories
MACRO_URL = '%smoney.finance.%s/mac/api/jsonp.php/SINAREMOTECALLCALLBACK%s/MacPage_Service.get_pagedata?cate=%s&event=%s&from=0&num=%s&condition=&_=%s'
GDP_YEAR_COLS = ['year','gdp','pc_gdp','gnp','pi','si','industry','cons_industry','ti','trans_industry','lbdy']
# 🧠 ML Signal: Usage of string formatting for constructing URLs
GDP_QUARTER_COLS = ['quarter','gdp','gdp_yoy','pi','pi_yoy','si','si_yoy','ti','ti_yoy']
# ⚠️ SAST Risk (Low): Potential for URL manipulation if inputs are not validated
GDP_FOR_COLS = ['year','end_for','for_rate','asset_for','asset_rate','goods_for','goods_rate']
GDP_PULL_COLS = ['year','gdp_yoy','pi','si','industry','ti']
# 🧠 ML Signal: Usage of list to define column names for GDP year data
GDP_CONTRIB_COLS = ['year','gdp_yoy','pi','si','industry','ti']
CPI_COLS = ['month','cpi']
# 🧠 ML Signal: Usage of list to define column names for GDP quarter data
PPI_COLS = ['month','ppiip','ppi','qm','rmi','pi','cg','food','clothing','roeu','dcg']
DEPOSIT_COLS = ['date','deposit_type','rate']
# 🧠 ML Signal: Usage of list to define column names for GDP foreign data
LOAN_COLS = ['date','loan_type','rate']
RRR_COLS = ['date','before','now','changed']
# 🧠 ML Signal: Usage of list to define column names for GDP pull data
MONEY_SUPPLY_COLS = ['month','m2','m2_yoy','m1','m1_yoy','m0','m0_yoy','cd','cd_yoy','qm','qm_yoy','ftd','ftd_yoy','sd','sd_yoy','rests','rests_yoy']
# ✅ Best Practice: Use a more descriptive function name to avoid confusion with the random module
MONEY_SUPPLY_BLA_COLS = ['year','m2','m1','m0','cd','qm','ftd','sd','rests']
# 🧠 ML Signal: Usage of list to define column names for GDP contribution data
GOLD_AND_FOREIGN_CURRENCY_RESERVES = ['month','gold','foreign_reserves']
# ✅ Best Practice: Import statements should be at the top of the file

# 🧠 ML Signal: Usage of list to define column names for CPI data
def random(n=13):
    # 🧠 ML Signal: Usage of list to define column names for PPI data
    # 🧠 ML Signal: Usage of list to define column names for money supply balance data
    # 🧠 ML Signal: Usage of list to define column names for gold and foreign currency reserves data
    # 🧠 ML Signal: Usage of power operator to calculate large numbers
    # 🧠 ML Signal: Conversion of integer to string
    from random import randint
    start = 10**(n-1)
    end = (10**n)-1
    return str(randint(start, end))