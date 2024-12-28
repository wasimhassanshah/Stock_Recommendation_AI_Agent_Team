from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.openbb_tools import OpenBBTools
from phi.tools.pandas import PandasTools
from phi.tools.calculator import Calculator
from phi.tools.csv_tools import CsvTools
from phi.tools.sql import SQLTools
from phi.tools.website import WebsiteTools
from phi.tools.wikipedia import WikipediaTools
from phi.tools.email import EmailTools
from phi.tools.slack import SlackTools
from phi.tools.github import GithubTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from yfinance import Ticker

# Load environment variables
load_dotenv()

# Define individual agents
def generate_plot_tool(ticker_symbol, title):
    ticker = Ticker(ticker_symbol)
    hist = ticker.history(period="6mo")
    plt.figure()
    plt.plot(hist.index, hist['Close'], label=ticker_symbol)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    file_path = f"{ticker_symbol}_plot.png"
    plt.savefig(file_path)
    plt.close()
    return file_path

research_agent = Agent(
    name="Research Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True), OpenBBTools()],
    instructions=["Use proper tabular format to display data."],
    show_tool_calls=True,
    markdown=True,
)

data_analysis_agent = Agent(
    name="Data Analysis Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_fundamentals=True), PandasTools(), Calculator()],
    instructions=["Analyze the data and provide insights."],
    show_tool_calls=True,
    markdown=True,
)

plotting_agent = Agent(
    name="Plotting Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[generate_plot_tool, CsvTools()],
    instructions=["Create visualizations based on the provided data. Ensure to save plots in the project directory."],
    show_tool_calls=True,
    markdown=True,
)

query_agent = Agent(
    name="Query Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[SQLTools()],
    instructions=["Interact with user queries effectively."],
    show_tool_calls=True,
    markdown=True,
)

recommendation_agent = Agent(
    name="Recommendation Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True), WebsiteTools()],
    instructions=["Provide recommendations on where to buy and sell stocks."],
    show_tool_calls=True,
    markdown=True,
)

news_agent = Agent(
    name="News Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo(), WikipediaTools()],
    instructions=["Summarize the latest news related to stocks and markets."],
    show_tool_calls=True,
    markdown=True,
)

portfolio_agent = Agent(
    name="Portfolio Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[EmailTools(), SlackTools()],
    instructions=["Provide advice on building and optimizing stock portfolios."],
    show_tool_calls=True,
    markdown=True,
)

risk_management_agent = Agent(
    name="Risk Management Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[GithubTools()],
    instructions=["Analyze risks and provide mitigation strategies for investments."],
    show_tool_calls=True,
    markdown=True,
)

sentiment_analysis_agent = Agent(
    name="Sentiment Analysis Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[],
    instructions=["Analyze market sentiment based on news and social media."],
    show_tool_calls=True,
    markdown=True,
)

# Define the multi-agent team
agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[
        research_agent,
        data_analysis_agent,
        plotting_agent,
        query_agent,
        recommendation_agent,
        news_agent,
        portfolio_agent,
        risk_management_agent,
        sentiment_analysis_agent
    ],
    instructions=["Always include sources", "Use tables to display data", "Coordinate among agents to handle complex queries."],
    show_tool_calls=True,
    markdown=True,
)

# Example usage of the agent team
query1 = (
    "Summarize and compare analyst recommendations for Apple and Microsoft, and should I buy or sell Apple stocks or not if yes how to? "
    "Can you plot the stock of both, and also tell what could be risk in buying stocks of both?"
)

# Collecting responses from the agent team
responses1 = agent_team.print_response(query1, stream=True)

# Additional queries can also be processed as needed
