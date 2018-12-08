#Campbell Boswell & Scott Westvold
#AIM - Automated Investment Manager
#cs701 - Senior Project
#constituents_filter.py

'''
A series of common words that are included in the names of corporations listed on
the S&P 500. These words do not count as keywords which can distinguish a Tweet
as relevant
'''

filter_list = {'a', 'at', 'holding', 're', 'm', 'worldwide', 'lines', 'service', 'restaurants', 'products', 'black', 'exchange', 'paper', 'information', 'trust', 'transport', 'line', 'water', 'texas', 'equity', 'mountain', 'noble', 'mining', 'homes', 'security', 'soup', 'union', 'depot', 'technologies', 'class', 'york', 'payments', 'cruise', 'robert', 'l', 'marathon', 'norwegian', 'air', 'new', 'news', 'extra', 'digital', 'street', 'host', 'one', 'federal', 'boston', 'waste', 'green', 'business', 'travelers', 'sachs', 'ca', 'c', 'system', 'capital', 'space', 'red', 'growth', 'corporation', 'news', 'century', 'west', 'foods', 'fifth', 'real', 'advance', 'alphabet', 'company', 'gas', 'national', 'electric', 'home', 'data', 'j', 'constellation', 'best', 'sl', 'systems', 'crown', 'te', 'gamble', 'total', 'progressive', 'group', 'pacific', 'affiliated', 'airlines', 'tool', 'health', 'royal', 'smith', 'p', 'services', 'apartment', 'apartments', 'michael', 'research', 'works', 'management', 'stores', 'state', 'markets', 'johnson', 'yum', 'third', 'investment', 'dynamics', 'global', 'fortune', 'american', 'bank', 'estate', 'communications', 'devices', 'quest', 'block', 'buy', 'income', 'energy', 'people', 'h', 'b', 'financial', 'companies', 'united', 'power', 'brands', 'network', 'time', 'international', 'parts', 'general', 'pinnacle', 'auto', 'mills', 'medical', 'industries', 'packaging', 'half', 'public', 'price', 'flavors', 'martin', 'express', 'r', 'america', 'dr',  'morgan', 'city', 'motors', 'beverage', 'market', 'james', 'dollar', 'fargo', 'e', 'campbell', 'hat', 'healthcare', 'church', 'kansas', 's', 'com', 'cruises', 'air', 'property', 'solutions', 'on', 'corp', 't', 'connectivity', 'inc', 'scientific', 'co', 'system', 'supply', 'u', 's', 'v', 'f', 'towers', 'resorts', 'holdings'}
