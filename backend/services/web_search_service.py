from __future__ import annotations

import html
import logging
import re
import urllib.parse
import urllib.request
from typing import Any

LOGGER = logging.getLogger(__name__)

BULLISH_KEYWORDS = {
    "surge", "rally", "gain", "gains", "jump", "jumps", "bullish", "high", "higher",
    "rise", "rising", "growth", "profit", "uptrend", "breakout", "buy", "upgraded",
    "outperform", "rebound", "positive", "record", "soar", "soaring", "strong",
}

BEARISH_KEYWORDS = {
    "fall", "falling", "drop", "drops", "slump", "decline", "declining", "bearish",
    "low", "lower", "loss", "losses", "downtrend", "breakdown", "sell", "downgraded",
    "underperform", "plunge", "plunging", "negative", "crash", "weak", "correction",
}


def clean_html_tags(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    clean = re.sub(r"<[^>]+>", "", text)
    return html.unescape(clean).strip()


class WebSearchService:
    """Service to retrieve web search context and market news sentiment for tickers/charts."""

    def __init__(self, timeout: float = 6.0) -> None:
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    def search_duckduckgo(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Search DuckDuckGo Lite / HTML for clean web results without API keys."""
        encoded_query = urllib.parse.quote(query)
        # Using DuckDuckGo Lite which is fast and lightweight
        url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
        results: list[dict[str, str]] = []

        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read().decode("utf-8", errors="ignore")

            # Extract result links and snippets from DuckDuckGo Lite HTML
            # Links have class 'result-link'
            link_matches = re.findall(
                r'<a[^>]+class=[\'"]result-link[\'"][^>]+href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>',
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            # Snippets have class 'result-snippet'
            snippet_matches = re.findall(
                r'<td[^>]+class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>',
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )

            for i in range(min(len(link_matches), max_results)):
                href, title_raw = link_matches[i]
                title = clean_html_tags(title_raw)
                snippet = clean_html_tags(snippet_matches[i]) if i < len(snippet_matches) else ""

                # Decode real URL if redirected through ddg
                if "uddg=" in href:
                    parsed_url = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    real_url = parsed_url.get("uddg", [href])[0]
                else:
                    real_url = href

                if title:
                    results.append({
                        "title": title,
                        "url": real_url,
                        "snippet": snippet,
                    })

        except Exception as exc:
            LOGGER.warning("DuckDuckGo search error for query '%s': %s", query, exc)

        return results

    def search_yahoo_rss(self, symbol: str, max_results: int = 4) -> list[dict[str, str]]:
        """Fetch latest financial headlines via Yahoo Finance RSS."""
        clean_symbol = re.sub(r"[^A-Za-z0-9\.\-=]", "", symbol.strip().upper())
        if not clean_symbol:
            return []

        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={clean_symbol}"
        results: list[dict[str, str]] = []

        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                xml_content = response.read().decode("utf-8", errors="ignore")

            items = re.findall(r"<item>(.*?)</item>", xml_content, flags=re.DOTALL)
            for item in items[:max_results]:
                title_match = re.search(r"<title>(.*?)</title>", item, flags=re.DOTALL)
                link_match = re.search(r"<link>(.*?)</link>", item, flags=re.DOTALL)
                desc_match = re.search(r"<description>(.*?)</description>", item, flags=re.DOTALL)

                title = clean_html_tags(title_match.group(1)) if title_match else ""
                link = clean_html_tags(link_match.group(1)) if link_match else ""
                snippet = clean_html_tags(desc_match.group(1)) if desc_match else ""

                if title:
                    results.append({
                        "title": title,
                        "url": link,
                        "snippet": snippet,
                    })
        except Exception as exc:
            LOGGER.info("Yahoo RSS not available for %s: %s", clean_symbol, exc)

        return results

    def analyze_sentiment(self, text: str) -> tuple[str, float]:
        """Compute keyword-based market sentiment from headlines and snippets."""
        words = re.findall(r"\b[A-Za-z]+\b", text.lower())
        bull_count = sum(1 for w in words if w in BULLISH_KEYWORDS)
        bear_count = sum(1 for w in words if w in BEARISH_KEYWORDS)
        total = bull_count + bear_count

        if total == 0:
            return "Neutral", 0.50

        bull_ratio = bull_count / total
        if bull_ratio >= 0.60:
            sentiment = "Bullish"
            score = round(0.50 + (bull_ratio * 0.45), 2)
        elif bull_ratio <= 0.40:
            sentiment = "Bearish"
            score = round(0.50 + ((1.0 - bull_ratio) * 0.45), 2)
        else:
            sentiment = "Neutral"
            score = 0.50

        return sentiment, score

    def search_and_analyze(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        predicted_direction: str | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive web search and produce sentiment + insight synthesis."""
        clean_symbol = symbol.strip().upper() if symbol and symbol.strip() else None
        timeframe_str = f" {timeframe}" if timeframe and timeframe.strip() else ""

        if clean_symbol:
            query = f"{clean_symbol} stock price technical analysis{timeframe_str} market forecast"
        elif predicted_direction:
            query = f"candlestick pattern {predicted_direction} technical analysis trading strategy"
        else:
            query = "candlestick chart pattern market technical analysis"

        # 1. Fetch search results
        results = []
        if clean_symbol:
            results.extend(self.search_yahoo_rss(clean_symbol, max_results=3))

        ddg_results = self.search_duckduckgo(query, max_results=5)
        for item in ddg_results:
            if not any(r["title"] == item["title"] for r in results):
                results.append(item)

        # 2. Analyze sentiment from aggregated titles and snippets
        aggregated_text = " ".join(f"{r['title']} {r['snippet']}" for r in results)
        sentiment, sentiment_score = self.analyze_sentiment(aggregated_text)

        # 3. Formulate synthesized insight
        if clean_symbol:
            summary = (
                f"Web market scan for {clean_symbol}{timeframe_str} indicates {sentiment.upper()} sentiment "
                f"({int(sentiment_score * 100)}% signal strength) based on {len(results)} live financial sources. "
            )
            if predicted_direction:
                if sentiment.lower() == predicted_direction.lower():
                    summary += (
                        f"Live web signals ALIGN with the detected {predicted_direction} candlestick pattern."
                    )
                elif sentiment != "Neutral":
                    summary += (
                        f"Note: Current news sentiment ({sentiment}) differs from chart pattern ({predicted_direction}); "
                        "exercise caution around high-impact volatility."
                    )
                else:
                    summary += (
                        f"Web sentiment is currently balanced, providing neutral macro context for the {predicted_direction} setup."
                    )
        else:
            summary = (
                f"Web search for {predicted_direction or 'candlestick'} setups retrieved {len(results)} technical sources. "
                f"Market commentary leans {sentiment}."
            )

        return {
            "query": query,
            "symbol": clean_symbol,
            "timeframe": timeframe,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "summary": summary,
            "results_count": len(results),
            "sources": results[:4],
        }
