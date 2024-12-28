from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define individual agents
research_agent = Agent(
    name="Research Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True)],
    instructions=["Use proper tabular format to display data."],
    show_tool_calls=True,
    markdown=True,
)

data_analysis_agent = Agent(
    name="Data Analysis Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_fundamentals=True)],
    instructions=["Analyze the data and provide insights."],
    show_tool_calls=True,
    markdown=True,
)

plotting_agent = Agent(
    name="Plotting Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[],  # Add plotting tools here if available
    instructions=["Create visualizations based on the provided data."],
    show_tool_calls=True,
    markdown=True,
)

query_agent = Agent(
    name="Query Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[],
    instructions=["Interact with user queries effectively."],
    show_tool_calls=True,
    markdown=True,
)

recommendation_agent = Agent(
    name="Recommendation Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True)],
    instructions=["Provide recommendations on where to buy and sell stocks."],
    show_tool_calls=True,
    markdown=True,
)

# Define the multi-agent team
agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[research_agent, data_analysis_agent, plotting_agent, query_agent, recommendation_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Example usage of the agent team
query1 = "Summarize and compare analyst recommendations for Apple and Microsoft, and should I buy or sell Apple stocks or not if yes how to?"
query2 = "Analyze the fundamentals of Apple and Microsoft"
query3 = "Plot the stock price trends for Apple and Microsoft"
query4 = "How can I invest in technology stocks?"
query5 = "Should I buy or sell Apple stocks? If yes, how?"

# Collecting responses from the agent team
responses1 = agent_team.print_response(query1, stream=True)

# Additional queries can also be processed as needed
