"""
Test script for all 5 Gemini Agent loop patterns
Tests each pattern independently to verify functionality
"""

import asyncio
import os
import sys
import time
from threading import Event
import numpy as np

# Import agent components
from gemini_agent import GeminiAgent, AgentMode, VisualContext, UserQuery, AgentResponse

# Check for API key
if not os.getenv('GOOGLE_API_KEY'):
    print("❌ GOOGLE_API_KEY not set in environment")
    print("   Set it in .env file or export GOOGLE_API_KEY=your_key")
    sys.exit(1)


# =============================================================================
# TEST HELPERS
# =============================================================================

def create_test_context(frame_id: int, has_obstacle: bool = False) -> VisualContext:
    """Create a test visual context"""
    if has_obstacle:
        detections = [
            {
                'class': 'person',
                'estimated_distance': 1.5,
                'confidence': 0.92,
                'center_x': 320,
                'center_y': 240
            }
        ]
        guidance = "⚠️ Person ahead, 1 meters"
    else:
        detections = []
        guidance = "Path clear, continue straight"

    return VisualContext(
        detections=detections,
        count=len(detections),
        guidance=guidance,
        timestamp=time.time(),
        frame_id=frame_id
    )


# =============================================================================
# PATTERN 1: Q&A LOOP TEST
# =============================================================================

async def test_qa_loop():
    """Test Pattern 1: Q&A Loop"""
    print("\n" + "=" * 70)
    print("🧪 TESTING PATTERN 1: Q&A LOOP")
    print("=" * 70)

    # Initialize agent
    agent = GeminiAgent(mode=AgentMode.QA)
    agent.start_conversation()

    # Update visual context
    context = create_test_context(1, has_obstacle=True)
    agent.update_visual_context(context)

    # Simulate user queries
    test_queries = [
        "What's in front of me?",
        "Is it safe to continue?",
        "How far away is the obstacle?"
    ]

    print("\n📝 Running test queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n  Query {i}: {query}")

        try:
            response = await agent.query(query, include_visual_context=True)
            print(f"  Response: {response.text[:100]}...")
            print(f"  ✅ Success")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n✅ Q&A Loop test complete")


# =============================================================================
# PATTERN 2: MONITORING LOOP TEST
# =============================================================================

async def test_monitoring_loop():
    """Test Pattern 2: Monitoring Loop"""
    print("\n" + "=" * 70)
    print("🧪 TESTING PATTERN 2: MONITORING LOOP")
    print("=" * 70)

    # Initialize agent
    agent = GeminiAgent(mode=AgentMode.MONITOR)
    agent.start_conversation()

    # Alert queue
    alerts_received = []

    async def send_alert(alert: AgentResponse):
        """Mock alert sender"""
        alerts_received.append(alert.text)
        print(f"  🔔 Alert: {alert.text}")

    # Stop event
    stop_event = Event()

    # Start monitoring loop
    print("\n📡 Starting monitoring loop (5 seconds)...")

    # Create monitoring task
    monitor_task = asyncio.create_task(
        agent.monitoring_loop(send_alert, stop_event, alert_interval=2.0)
    )

    # Simulate significant changes
    await asyncio.sleep(1)
    context1 = create_test_context(1, has_obstacle=True)
    agent.update_visual_context(context1)
    agent.last_significant_change = context1
    print("  ✅ Simulated obstacle detected")

    # Wait for alerts
    await asyncio.sleep(3)

    # Stop monitoring
    stop_event.set()
    await asyncio.sleep(0.5)

    print(f"\n📊 Alerts received: {len(alerts_received)}")
    print("✅ Monitoring Loop test complete")


# =============================================================================
# PATTERN 3: NAVIGATION LOOP TEST
# =============================================================================

async def test_navigation_loop():
    """Test Pattern 3: Navigation Loop"""
    print("\n" + "=" * 70)
    print("🧪 TESTING PATTERN 3: NAVIGATION LOOP")
    print("=" * 70)

    # Initialize agent
    agent = GeminiAgent(mode=AgentMode.NAVIGATION)
    agent.start_conversation()

    # Guidance queue
    guidance_received = []

    async def send_guidance(guidance: AgentResponse):
        """Mock guidance sender"""
        guidance_received.append(guidance.text)
        print(f"  🧭 Guidance: {guidance.text}")

    # Stop event
    stop_event = Event()

    # Start navigation loop
    print("\n🧭 Starting navigation loop (5 seconds)...")

    # Create navigation task
    nav_task = asyncio.create_task(
        agent.navigation_loop(send_guidance, stop_event, guidance_interval=2.0)
    )

    # Simulate frame updates
    for i in range(3):
        await asyncio.sleep(1.5)
        context = create_test_context(i + 1, has_obstacle=(i % 2 == 0))
        agent.update_visual_context(context)
        print(f"  ✅ Frame {i + 1} updated")

    # Stop navigation
    stop_event.set()
    await asyncio.sleep(0.5)

    print(f"\n📊 Guidance messages: {len(guidance_received)}")
    print("✅ Navigation Loop test complete")


# =============================================================================
# PATTERN 4: CONTEXTUAL LOOP TEST
# =============================================================================

async def test_contextual_loop():
    """Test Pattern 4: Contextual Awareness Loop"""
    print("\n" + "=" * 70)
    print("🧪 TESTING PATTERN 4: CONTEXTUAL LOOP")
    print("=" * 70)

    # Initialize agent
    agent = GeminiAgent(mode=AgentMode.CONTEXTUAL)
    agent.start_conversation()

    # Description queue
    descriptions_received = []

    async def send_description(description: AgentResponse):
        """Mock description sender"""
        descriptions_received.append(description.text)
        print(f"  📝 Description: {description.text[:100]}...")

    # Stop event
    stop_event = Event()

    # Start contextual loop
    print("\n📝 Starting contextual loop (8 seconds)...")

    # Create contextual task
    context_task = asyncio.create_task(
        agent.contextual_loop(send_description, stop_event, description_interval=3.0)
    )

    # Simulate significant scene change
    await asyncio.sleep(1)
    context1 = create_test_context(1, has_obstacle=True)
    agent.update_visual_context(context1)
    agent.last_significant_change = context1
    print("  ✅ Significant scene change detected")

    # Wait for description
    await asyncio.sleep(5)

    # Stop contextual loop
    stop_event.set()
    await asyncio.sleep(0.5)

    print(f"\n📊 Descriptions generated: {len(descriptions_received)}")
    print("✅ Contextual Loop test complete")


# =============================================================================
# PATTERN 5: HYBRID LOOP TEST
# =============================================================================

async def test_hybrid_loop():
    """Test Pattern 5: Hybrid Loop (Navigation + Q&A)"""
    print("\n" + "=" * 70)
    print("🧪 TESTING PATTERN 5: HYBRID LOOP")
    print("=" * 70)

    # Initialize agent
    agent = GeminiAgent(mode=AgentMode.HYBRID)
    agent.start_conversation()

    # Query queue (simulates user queries)
    query_queue = asyncio.Queue()

    # Response queue
    responses_received = []

    async def get_user_query():
        """Mock query getter"""
        try:
            query = await asyncio.wait_for(query_queue.get(), timeout=0.1)
            return query
        except asyncio.TimeoutError:
            return None

    async def send_response(response: AgentResponse):
        """Mock response sender"""
        responses_received.append(response.text)
        if hasattr(response, 'mode'):
            print(f"  💬 [{response.mode.value}] {response.text[:80]}...")
        else:
            print(f"  💬 {response.text[:80]}...")

    # Stop event
    stop_event = Event()

    # Start hybrid loop
    print("\n🔀 Starting hybrid loop (8 seconds)...")

    # Create hybrid task
    hybrid_task = asyncio.create_task(
        agent.hybrid_loop(get_user_query, send_response, stop_event, navigation_interval=3.0)
    )

    # Simulate navigation updates
    for i in range(2):
        await asyncio.sleep(2)
        context = create_test_context(i + 1, has_obstacle=False)
        agent.update_visual_context(context)
        print(f"  ✅ Frame {i + 1} updated (auto-nav)")

    # Simulate user query
    await asyncio.sleep(1)
    user_query = UserQuery(
        audio_data=b'',
        text="What do you see?"
    )
    await query_queue.put(user_query)
    print("  ✅ User query sent")

    # Wait for response
    await asyncio.sleep(3)

    # Stop hybrid loop
    stop_event.set()
    await asyncio.sleep(0.5)

    print(f"\n📊 Responses received: {len(responses_received)}")
    print("✅ Hybrid Loop test complete")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    """Run all pattern tests"""
    print("\n" + "=" * 70)
    print("🚀 GEMINI AGENT - LOOP PATTERN TEST SUITE")
    print("=" * 70)

    tests = [
        ("Q&A Loop", test_qa_loop),
        ("Monitoring Loop", test_monitoring_loop),
        ("Navigation Loop", test_navigation_loop),
        ("Contextual Loop", test_contextual_loop),
        ("Hybrid Loop", test_hybrid_loop)
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
            print(f"\n✅ {name} PASSED")
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

        # Pause between tests
        print("\n" + "-" * 70)
        await asyncio.sleep(2)

    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    print("=" * 70)


if __name__ == "__main__":
    print("🔧 Initializing test environment...")

    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment")

    # Check API key again
    if not os.getenv('GOOGLE_API_KEY'):
        print("\n❌ GOOGLE_API_KEY still not set!")
        print("   Make sure to:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your Google API key to .env")
        print("   3. Run this test again")
        sys.exit(1)

    print("✅ API key found")

    # Run tests
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
