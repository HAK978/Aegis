"""
Gemini Live API Agent Integration for Visual Guidance System
Implements 5 loop patterns for different user interaction modes

Patterns:
1. Q&A Loop: User asks questions, agent responds with visual context
2. Monitoring Loop: Continuous environmental awareness alerts
3. Navigation Loop: Real-time turn-by-turn guidance
4. Contextual Loop: Proactive scene description on significant changes
5. Hybrid Loop: Combines navigation + Q&A for flexible interaction
"""

import asyncio
import os
import sys
import base64
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import threading

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed")
    print("   Install with: pip install google-generativeai>=0.8.3")

from audio_processor import AudioProcessor


# =============================================================================
# AGENT MODES
# =============================================================================

class AgentMode(Enum):
    """Agent operational modes"""
    QA = "agent_qa"                      # Q&A: User asks, agent answers
    MONITOR = "agent_monitor"            # Monitoring: Proactive alerts
    NAVIGATION = "agent_nav"             # Navigation: Turn-by-turn guidance
    CONTEXTUAL = "agent_contextual"      # Contextual: Smart scene awareness
    HYBRID = "agent_hybrid"              # Hybrid: Navigation + Q&A


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VisualContext:
    """Visual context from object detection system"""
    detections: List[Dict[str, Any]]
    count: int
    guidance: str
    timestamp: float
    frame_id: int = 0

    def to_text(self) -> str:
        """Convert to text description for LLM"""
        if self.count == 0:
            return "No objects detected. Path appears clear."

        # Build structured description
        lines = [f"Scene Analysis (Frame {self.frame_id}):"]
        lines.append(f"- Objects detected: {self.count}")
        lines.append(f"- Navigation guidance: {self.guidance}")
        lines.append("\nObject details:")

        for i, obj in enumerate(self.detections[:10], 1):  # Limit to top 10
            distance = obj.get('estimated_distance', 'unknown')
            confidence = obj.get('confidence', 0) * 100
            obj_class = obj.get('class', 'unknown')

            lines.append(
                f"  {i}. {obj_class} - {distance}m away (confidence: {confidence:.0f}%)"
            )

        if len(self.detections) > 10:
            lines.append(f"  ... and {len(self.detections) - 10} more objects")

        return "\n".join(lines)


@dataclass
class UserQuery:
    """User voice query"""
    audio_data: bytes
    text: Optional[str] = None  # If using STT
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class AgentResponse:
    """Agent response"""
    text: str
    audio_data: Optional[bytes] = None
    timestamp: float = 0.0
    mode: AgentMode = AgentMode.QA

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# =============================================================================
# GEMINI AGENT
# =============================================================================

class GeminiAgent:
    """
    Gemini Live API Agent for Visual Guidance System
    Supports 5 different loop patterns for diverse user interactions
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gemini-2.0-flash-exp",  # Default, can override with gemini-2.5-flash-latest
                 mode: AgentMode = AgentMode.HYBRID):
        """
        Initialize Gemini Agent

        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model: Gemini model to use (gemini-2.5-flash-latest recommended)
            mode: Agent operational mode
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")

        # API Configuration
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=self.api_key)

        # Model Configuration
        self.model_name = model
        self.model = genai.GenerativeModel(model)
        self.chat = None  # Will be initialized when starting conversation

        # Agent State
        self.mode = mode
        self.is_running = False
        self.audio_processor = AudioProcessor(enable_vad=True)

        # Context Management
        self.visual_context_history = deque(maxlen=10)  # Last 10 frames
        self.conversation_history = deque(maxlen=50)  # Last 50 exchanges
        self.last_significant_change = None

        # Rate Limiting
        self.last_query_time = 0
        self.min_query_interval = 1.0  # Minimum seconds between queries

        print(f"✅ Gemini Agent initialized")
        print(f"   Model: {self.model_name}")
        print(f"   Mode: {self.mode.value}")

    # =========================================================================
    # CORE METHODS
    # =========================================================================

    def start_conversation(self, system_instruction: Optional[str] = None):
        """
        Start a new conversation session

        Args:
            system_instruction: Custom system instruction for the agent
        """
        if system_instruction is None:
            system_instruction = self._get_system_instruction_for_mode()

        # Start chat with system instruction
        self.chat = self.model.start_chat(history=[])
        self.conversation_history.clear()

        print(f"✅ Conversation started in {self.mode.value} mode")

    def update_visual_context(self, context: VisualContext):
        """
        Update visual context from detection system

        Args:
            context: New visual context
        """
        self.visual_context_history.append(context)

        # Detect significant changes for contextual mode
        if len(self.visual_context_history) >= 2:
            prev = self.visual_context_history[-2]
            curr = context

            # Significant change: new objects, object count change >50%, critical guidance
            obj_change = abs(curr.count - prev.count) / max(prev.count, 1)
            critical_keywords = ['STOP', 'CAUTION', 'WARNING', '⚠️']
            has_critical = any(kw in curr.guidance.upper() for kw in critical_keywords)

            if obj_change > 0.5 or has_critical:
                self.last_significant_change = context

    async def query(self,
                    user_input: str,
                    include_visual_context: bool = True) -> AgentResponse:
        """
        Send a text query to the agent

        Args:
            user_input: User's question/command
            include_visual_context: Whether to include current visual context

        Returns:
            AgentResponse with text and optional audio
        """
        # Rate limiting
        time_since_last = time.time() - self.last_query_time
        if time_since_last < self.min_query_interval:
            await asyncio.sleep(self.min_query_interval - time_since_last)

        try:
            # Build context-aware prompt
            prompt = self._build_prompt(user_input, include_visual_context)

            # Send to Gemini
            response = await asyncio.to_thread(
                self.chat.send_message,
                prompt
            )

            response_text = response.text

            # Track conversation
            self.conversation_history.append({
                'user': user_input,
                'agent': response_text,
                'timestamp': time.time()
            })

            self.last_query_time = time.time()

            return AgentResponse(
                text=response_text,
                mode=self.mode
            )

        except Exception as e:
            print(f"❌ Query error: {e}")
            return AgentResponse(
                text=f"Sorry, I encountered an error: {str(e)}",
                mode=self.mode
            )

    # =========================================================================
    # LOOP PATTERN 1: Q&A LOOP
    # =========================================================================

    async def qa_loop(self,
                      get_user_query: Callable,
                      send_response: Callable,
                      stop_event: threading.Event):
        """
        Pattern 1: Question & Answer Loop
        User asks questions, agent responds with visual context

        Args:
            get_user_query: Async function that returns UserQuery or None
            send_response: Async function to send AgentResponse
            stop_event: Threading event to signal stop
        """
        print("🎯 Starting Q&A Loop")
        self.start_conversation()

        while not stop_event.is_set():
            try:
                # Wait for user query
                user_query = await get_user_query()

                if user_query is None:
                    await asyncio.sleep(0.1)
                    continue

                # Process query (text or audio)
                if user_query.text:
                    query_text = user_query.text
                else:
                    # TODO: Add STT if needed
                    query_text = "[Audio query - transcription not available]"

                # Get response with visual context
                response = await self.query(query_text, include_visual_context=True)

                # Send to user
                await send_response(response)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Q&A loop error: {e}")
                await asyncio.sleep(1)

        print("🛑 Q&A Loop stopped")

    # =========================================================================
    # LOOP PATTERN 2: MONITORING LOOP
    # =========================================================================

    async def monitoring_loop(self,
                             send_alert: Callable,
                             stop_event: threading.Event,
                             alert_interval: float = 5.0):
        """
        Pattern 2: Continuous Monitoring Loop
        Proactively alerts user of environmental changes

        Args:
            send_alert: Async function to send AgentResponse alerts
            stop_event: Threading event to signal stop
            alert_interval: Minimum seconds between alerts
        """
        print("🎯 Starting Monitoring Loop")
        self.start_conversation()

        last_alert_time = 0

        while not stop_event.is_set():
            try:
                current_time = time.time()

                # Check if significant change occurred
                if self.last_significant_change:
                    time_since_alert = current_time - last_alert_time

                    if time_since_alert >= alert_interval:
                        # Generate alert
                        context = self.last_significant_change
                        alert_text = f"Alert: {context.guidance}"

                        # Get AI-enhanced explanation if critical
                        if '⚠️' in context.guidance or 'STOP' in context.guidance:
                            prompt = f"The visual system detected: {context.to_text()}\n\nProvide a brief, urgent warning to a blind user."
                            response = await self.query(prompt, include_visual_context=False)
                            alert_text = response.text

                        # Send alert
                        alert = AgentResponse(
                            text=alert_text,
                            mode=AgentMode.MONITOR
                        )
                        await send_alert(alert)

                        last_alert_time = current_time
                        self.last_significant_change = None  # Clear after alerting

                await asyncio.sleep(0.5)  # Check every 500ms

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(1)

        print("🛑 Monitoring Loop stopped")

    # =========================================================================
    # LOOP PATTERN 3: NAVIGATION LOOP
    # =========================================================================

    async def navigation_loop(self,
                             send_guidance: Callable,
                             stop_event: threading.Event,
                             guidance_interval: float = 2.0):
        """
        Pattern 3: Real-time Navigation Loop
        Continuous turn-by-turn guidance based on visual context

        Args:
            send_guidance: Async function to send navigation guidance
            stop_event: Threading event to signal stop
            guidance_interval: Seconds between guidance updates
        """
        print("🎯 Starting Navigation Loop")
        self.start_conversation()

        last_guidance_time = 0

        while not stop_event.is_set():
            try:
                current_time = time.time()
                time_since_guidance = current_time - last_guidance_time

                if time_since_guidance >= guidance_interval:
                    # Get latest visual context
                    if self.visual_context_history:
                        context = self.visual_context_history[-1]

                        # Send basic guidance (from vision system)
                        guidance = AgentResponse(
                            text=context.guidance,
                            mode=AgentMode.NAVIGATION
                        )
                        await send_guidance(guidance)

                        last_guidance_time = current_time

                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Navigation loop error: {e}")
                await asyncio.sleep(1)

        print("🛑 Navigation Loop stopped")

    # =========================================================================
    # LOOP PATTERN 4: CONTEXTUAL LOOP
    # =========================================================================

    async def contextual_loop(self,
                             send_description: Callable,
                             stop_event: threading.Event,
                             description_interval: float = 10.0):
        """
        Pattern 4: Contextual Awareness Loop
        Proactive scene descriptions on significant visual changes

        Args:
            send_description: Async function to send scene descriptions
            stop_event: Threading event to signal stop
            description_interval: Minimum seconds between descriptions
        """
        print("🎯 Starting Contextual Loop")
        self.start_conversation()

        last_description_time = 0

        while not stop_event.is_set():
            try:
                current_time = time.time()

                # Describe scene on significant changes or interval
                if self.last_significant_change:
                    time_since_desc = current_time - last_description_time

                    if time_since_desc >= description_interval:
                        context = self.last_significant_change

                        # Generate AI description
                        prompt = f"{context.to_text()}\n\nDescribe this scene to a blind user in 2-3 sentences. Focus on what's most important for navigation."
                        response = await self.query(prompt, include_visual_context=False)

                        description = AgentResponse(
                            text=response.text,
                            mode=AgentMode.CONTEXTUAL
                        )
                        await send_description(description)

                        last_description_time = current_time
                        self.last_significant_change = None

                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Contextual loop error: {e}")
                await asyncio.sleep(1)

        print("🛑 Contextual Loop stopped")

    # =========================================================================
    # LOOP PATTERN 5: HYBRID LOOP
    # =========================================================================

    async def hybrid_loop(self,
                         get_user_query: Callable,
                         send_response: Callable,
                         stop_event: threading.Event,
                         navigation_interval: float = 3.0):
        """
        Pattern 5: Hybrid Loop (Navigation + Q&A)
        Combines continuous navigation with on-demand Q&A

        Args:
            get_user_query: Async function that returns UserQuery or None
            send_response: Async function to send AgentResponse
            stop_event: Threading event to signal stop
            navigation_interval: Seconds between auto-navigation updates
        """
        print("🎯 Starting Hybrid Loop (Navigation + Q&A)")
        self.start_conversation()

        last_nav_time = 0

        while not stop_event.is_set():
            try:
                current_time = time.time()

                # Check for user query (priority)
                user_query = await get_user_query()

                if user_query:
                    # Handle Q&A
                    if user_query.text:
                        query_text = user_query.text
                    else:
                        query_text = "[Audio query]"

                    response = await self.query(query_text, include_visual_context=True)
                    await send_response(response)

                    # Reset navigation timer after interaction
                    last_nav_time = current_time

                else:
                    # Auto-navigation mode
                    time_since_nav = current_time - last_nav_time

                    if time_since_nav >= navigation_interval:
                        if self.visual_context_history:
                            context = self.visual_context_history[-1]

                            # Send navigation guidance
                            guidance = AgentResponse(
                                text=context.guidance,
                                mode=AgentMode.HYBRID
                            )
                            await send_response(guidance)

                            last_nav_time = current_time

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Hybrid loop error: {e}")
                await asyncio.sleep(1)

        print("🛑 Hybrid Loop stopped")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_system_instruction_for_mode(self) -> str:
        """Get system instruction based on agent mode"""
        base = """You are an AI assistant helping a blind or visually impaired person navigate their environment.
You receive real-time visual data from a camera with object detection and distance estimation.
Your responses should be:
- Clear and concise
- Actionable and immediate
- Safety-focused
- Empathetic but not patronizing
"""

        mode_specific = {
            AgentMode.QA: "\nMode: Q&A - Answer user questions about their surroundings using visual context.",
            AgentMode.MONITOR: "\nMode: Monitoring - Proactively alert users of important environmental changes.",
            AgentMode.NAVIGATION: "\nMode: Navigation - Provide turn-by-turn guidance for safe movement.",
            AgentMode.CONTEXTUAL: "\nMode: Contextual - Describe scenes when significant changes occur.",
            AgentMode.HYBRID: "\nMode: Hybrid - Combine navigation guidance with Q&A capabilities."
        }

        return base + mode_specific.get(self.mode, "")

    def _build_prompt(self, user_input: str, include_visual: bool) -> str:
        """Build context-aware prompt"""
        parts = []

        # Add visual context if requested
        if include_visual and self.visual_context_history:
            latest_context = self.visual_context_history[-1]
            parts.append("CURRENT VISUAL CONTEXT:")
            parts.append(latest_context.to_text())
            parts.append("\n")

        # Add user input
        parts.append("USER QUERY:")
        parts.append(user_input)

        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'mode': self.mode.value,
            'conversation_turns': len(self.conversation_history),
            'visual_context_frames': len(self.visual_context_history),
            'is_running': self.is_running,
            'last_query_time': self.last_query_time
        }


# =============================================================================
# TESTING & DEMO
# =============================================================================

async def test_agent():
    """Test the Gemini Agent"""
    print("=" * 70)
    print("🧪 TESTING GEMINI AGENT")
    print("=" * 70)

    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not set")
        print("   Set it in .env file or environment variable")
        return

    # Initialize agent
    print("\n📦 Initializing agent...")
    agent = GeminiAgent(mode=AgentMode.QA)

    # Create test visual context
    print("\n📸 Creating test visual context...")
    test_context = VisualContext(
        detections=[
            {
                'class': 'person',
                'estimated_distance': 2.5,
                'confidence': 0.92,
                'center_x': 320,
                'center_y': 240
            },
            {
                'class': 'car',
                'estimated_distance': 5.0,
                'confidence': 0.88,
                'center_x': 500,
                'center_y': 300
            }
        ],
        count=2,
        guidance="Person ahead, 2 meters. Car on right, 5 meters.",
        timestamp=time.time(),
        frame_id=1
    )

    agent.update_visual_context(test_context)

    # Test query
    print("\n💬 Testing query with visual context...")
    agent.start_conversation()

    response = await agent.query(
        "What's in front of me?",
        include_visual_context=True
    )

    print(f"\n✅ Response received:")
    print(f"   {response.text}")

    # Test stats
    print("\n📊 Agent stats:")
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ Agent test complete!")
    print("=" * 70)


if __name__ == "__main__":
    if not GEMINI_AVAILABLE:
        print("❌ Cannot run test: google-generativeai not installed")
        sys.exit(1)

    asyncio.run(test_agent())
