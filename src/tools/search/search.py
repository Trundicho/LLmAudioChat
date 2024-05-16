import re
from html import unescape
from urllib.parse import unquote

import orjson
import torch
from curl_cffi import requests as curl_requests
from sentence_transformers import util

from src.audio_chat_config import AudioChatConfig


class DuckDuckGoSearch:

    def __init__(self, embeddings_client):
        self.query_result = None
        self.asession = curl_requests.Session(impersonate="chrome", allow_redirects=False)
        config = AudioChatConfig().get_config()
        self.asession.headers["Referer"] = config["API_ENDPOINTS"]["DUCKDUCKGO"]
        self.embeddings_client = embeddings_client

    def _get_url(self, method, url, data):
        try:
            resp = self.asession.request(method, url, data=data)
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == (202, 301, 403):
                raise Exception(f"Error: {resp.status_code} rate limit error")
            if not resp:
                return None
        except Exception as error:
            if "timeout" in str(error).lower():
                raise TimeoutError("Duckduckgo timed out error")

    def duck(self, query, max_results=5, language="de-de", search_type=""):
        query1 = {"q": query}
        if "videos" == search_type:
            query1 = {"q": query, "ia": "videos", "iax": "videos", "iar": "videos"}
        resp = self._get_url("POST", "https://duckduckgo.com/", data=query1)
        vqd = self.extract_vqd(resp)

        number_of_pages = {
            5: "0",
            50: "",
        }
        params = {"q": query, "kl": language, "p": "1", "s": number_of_pages.get(max_results, "Invalid number"),
                  "df": "", "vqd": vqd, "ex": "", "m": f"{max_results}"}
        resp = self._get_url("GET", "https://links.duckduckgo.com/d.js", params)
        page_data = self.text_extract_json(resp)

        results = []
        for row in page_data:
            href = row.get("u")
            if href and href != f"http://www.google.com/search?q={query}":
                body = self.normalize(row["a"])
                if body:
                    result = {
                        "title": self.normalize(row["t"]),
                        "href": self.normalize_url(href),
                        "body": self.normalize(row["a"]),
                        "tags": self.normalize(row["da"]),
                    }
                    if not self.filter_search_result(result):
                        results.append(result)

        self.query_result = results
        return results

    def extract_youtube_results(self, results, user_input, top_k=3):
        extracted_data = []
        for item in results:
            if isinstance(item, dict):  # Ensure it's a dictionary
                if (item.get('href', '').startswith('https://www.youtube.com/watch')
                        or item.get('href', '').startswith('https://m.youtube.com/watch')):
                    title = item.get('title')
                    href = item.get('href')
                    extracted_data.append((title, href))

        if len(extracted_data) > 0:
            extracted_data_embedding = self.get_embedding(extracted_data)
            input_embedding = self.get_embedding([user_input])
            cos_scores = util.cos_sim(input_embedding, extracted_data_embedding)[0]
            top_k = min(top_k, len(cos_scores))
            top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
            relevant_context = [extracted_data[idx] for idx in top_indices]
            if len(relevant_context) > 0:
                return relevant_context[0]
        return None

    def get_embedding(self, data_array):
        return torch.tensor(self.embeddings_client.create_embeddings(data_array))

    def filter_search_result(self, result):
        return result.get("tags").__contains__(
            "mlb_games,nba_games,ncaafb_games,ncaamb_games,nfl_games,nhl_games,soccer_games,videos,wheretowatch")

    def duck_videos(self, query, max_results=3, language="de-de"):
        resp = self._get_url("POST", "https://duckduckgo.com/", data={"q": query, "ia": "videos", "iax": "videos",
                                                                      "iar": "videos"})
        vqd = self.extract_vqd(resp)

        number_of_pages = {
            5: "0",
            50: "",
        }
        params = {"q": query, "kl": language, "p": "1", "s": number_of_pages.get(5, "Invalid number"),
                  "df": "", "vqd": vqd, "ex": "", "m": f"{max_results}"}
        resp = self._get_url("GET", "https://links.duckduckgo.com/d.js", params)
        page_data = self.text_extract_json(resp)

        results = []
        for row in page_data:
            href = row.get("u")
            if href and href != f"http://www.google.com/search?q={query}":
                body = self.normalize(row["a"])
                if body:
                    result = {
                        "title": self.normalize(row["t"]),
                        "href": self.normalize_url(href),
                        "body": self.normalize(row["a"]),
                    }
                    results.append(result)

        self.query_result = results
        return results

    def search(self, query):
        self.duck(query)

    def get_first_link(self):
        return self.query_result[0]["href"]

    @staticmethod
    def extract_vqd(html_bytes: bytes) -> str:
        patterns = [(b'vqd="', 5, b'"'), (b"vqd=", 4, b"&"), (b"vqd='", 5, b"'")]
        for start_pattern, offset, end_pattern in patterns:
            try:
                start = html_bytes.index(start_pattern) + offset
                end = html_bytes.index(end_pattern, start)
                return html_bytes[start:end].decode()
            except ValueError:
                continue

    @staticmethod
    def text_extract_json(html_bytes):
        try:
            start = html_bytes.index(b"DDG.pageLayout.load('d',") + 24
            end = html_bytes.index(b");DDG.duckbar.load(", start)
            return orjson.loads(html_bytes[start:end])
        except Exception as ex:
            print(f"Error extracting JSON: {type(ex).__name__}: {ex}")

    @staticmethod
    def normalize_url(url: str) -> str:
        return unquote(url.replace(" ", "+")) if url else ""

    @staticmethod
    def normalize(raw_html: str) -> str:
        return unescape(re.sub("<.*?>", "", raw_html)) if raw_html else ""
