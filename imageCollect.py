# IMPORT
import os
import tweepy
import time
import urllib.request, urllib.error
from detect import detection_number
from conf import CK,CS,AK,AS

# 画像の保存先
IMG_DIR = 'C:/Users/kosakae256/OneDrive/Docments/CollectNumbers/tempimgs/'
detectpath = "C:/Users/kosakae256/OneDrive/Docments/CollectNumbers/detectimgs"
formpath = "C:/Users/kosakae256/OneDrive/Docments/CollectNumbers/formimgs"
jsonpath = "C:/Users/kosakae256/OneDrive/Docments/CollectNumbers/detectjsons/"

# 環境変数
CONSUMER_KEY        = CK
CONSUMER_SECRET     = CS
ACCESS_TOKEN_KEY    = AK
ACCESS_TOKEN_SECRET = AS

# 検索キーワード
TARGETS = ['#納車',"#プリウス","#ベンツ","#日産","#消防車","#パトカー","#コペン","#ホンダ","#ニッサン","#トヨタ","#ジープ","#メルセデス"]

# 検索オプション
SEARCH_PAGES_NUMBER = 10000 # 読み込むページ数
PER_PAGE_NUMBER = 100 # ページごとに返されるツイートの数（最大100）

class imageDownloader(object):
    def __init__(self):
        """初期設定
        """
        super(imageDownloader, self).__init__()
        self.set_api()

    def run(self):
        """実行
            1. twitterページを指定数取得
            2. ページ内のツイートのうち、キーワードがあるtweetのみ取得
            3. 画像URLを取得
            4. ダウンロード実行
            5. 判別
        """
        for target in TARGETS:
            self.max_id = None # ページを跨ぐ検索対象IDの初期化
            for page in range(SEARCH_PAGES_NUMBER):
                ret_url_list = self.search(target, PER_PAGE_NUMBER)

                #エラーはいた場合
                if ret_url_list == None:
                    time.sleep(60)
                    continue

                #検索しきった場合
                if ret_url_list == []:
                    break

                for url in ret_url_list:
                    print('OK ' + url)
                    self.download(url)
                    detection_number(self.path,detectpath,formpath,jsonpath,self.filename)

                time.sleep(1) # TimeOut防止

    def set_api(self):
        """apiの設定
        """
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(auth)

    def search(self, target, rpp):
        """twitterで検索実行
        """
        # 検索結果
        ret_url_list = []

        try:
            # 検索実行
            if self.max_id:
                # q: クエリ文字列, rpp: ツイート数, max_id: より小さい（古い）IDを持つステータスのみを返す
                res_search = self.api.search(q=target, lang='ja', rpp=rpp, max_id=self.max_id)
            else:
                res_search = self.api.search(q=target, lang='ja', rpp=rpp)
            # 結果を保存
            for result in res_search:
                if 'media' not in result.entities: continue
                for media in result.entities['media']:
                    url = media['media_url_https']
                    if url not in ret_url_list: ret_url_list.append(url)
            # 検索済みidの更新し、より古いツイートを検索させる
            self.max_id = result.id
            # 検索結果の返却
            return ret_url_list
        except Exception as e:
            self.error_catch(e)

    def download(self, url):
        """画像のダウンロード
        """
        url_orig = '%s:orig' % url
        self.path = IMG_DIR + url.split('/')[-1]
        self.filename = url.split('/')[-1]
        try:
            response = urllib.request.urlopen(url=url_orig)
            with open(self.path, "wb") as f:
                f.write(response.read())
        except Exception as e:
            self.error_catch(e)

    def error_catch(self, error):
        """エラー処理
        """
        print("NG ", error)

def main():
    """メイン処理
    """
    try:
        downloader = imageDownloader()
        downloader.run()
    except KeyboardInterrupt:
        # Ctrl-Cで終了
        pass

if __name__ == '__main__':
    main()
