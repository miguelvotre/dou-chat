"""
Arize Phoenix setup for production observability.

Automatically captures traces of LLM calls via OpenTelemetry.
Dashboard: https://app.phoenix.arize.com
"""

import os


def init_phoenix() -> None:
    """
    Initializes Arize Phoenix tracing.
    Should be called once at application startup.
    """
    import phoenix as px
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com"
    )
    api_key = os.getenv("PHOENIX_API_KEY")

    exporter = OTLPSpanExporter(
        endpoint=f"{endpoint}/v1/traces",
        headers={"api_key": api_key} if api_key else {},
    )

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    GoogleGenAIInstrumentor().instrument(tracer_provider=provider)
