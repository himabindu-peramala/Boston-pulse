"""
Boston Pulse — FastAPI Backend
Main entry point. Registers all routes.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes import health, chat, ingest

app = FastAPI(
    title="Boston Pulse Chatbot API",
    description="RAG-powered civic intelligence chatbot for Boston.",
    version="0.1.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,  tags=["Health"])
app.include_router(chat.router,    prefix="/api", tags=["Chat"])
app.include_router(ingest.router,  prefix="/api", tags=["Ingest"])
