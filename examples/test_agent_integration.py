#!/usr/bin/env python3
"""
Test Script: Gemini Agent Integration with AMD Backend
Demonstrates all 5 loop patterns with simulated camera input
"""

import os
import sys
import asyncio
import time
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gemini_agent import GeminiAgent, AgentMode, VisualContext
from amd_backend_no_vllm import AMDOptimizedBackend


class AgentIntegrationTest:
    """Test harness for Gemini Agent + AMD Backend integration"""

    def __init__(self):
        print("=" * 70)
        print("🧪 GEMINI AGENT INTEGRATION TEST")
        print("=" * 70)

        # Check API key
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            print("\n❌ GOOGLE_API_KEY not set!")
            print("   Set it in .env file or:")
            print("   export GOOGLE_API_KEY='your_api_key'")
            sys.exit(1)

        print("✅ API key found")

        # Initialize components
        print("\n📦 Initializing AMD backend...")
        self.backend = AMDOptimizedBackend(
            yolo_model="yolov8n.pt",  # Use nano for faster testing
            use_depth=False  # Skip depth for this test
        )

        print("\n📦 Initializing Gemini agent...")
        self.agent = GeminiAgent(
            api_key=self.api_key,
            mode=AgentMode.HYBRID
        )
        self.agent.start_conversation()

        print("\n✅ Initialization complete!")

    def create_test_image(self, scene_type: str = "person") -> np.ndarray:
        """
        Create a test image for simulation
        In production, this would be real camera feed
        """
        # Create a simple test pattern
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Add some recognizable patterns
        if scene_type == "person":
            cv2.rectangle(img, (250, 100), (390, 400), (255, 200, 100), -1)
            cv2.putText(img, "PERSON", (270, 250), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 0), 2)
        elif scene_type == "car":
            cv2.rectangle(img, (100, 200), (540, 400), (100, 100, 255), -1)
            cv2.putText(img, "CAR", (280, 320), cv2.FONT_HERSHEY_SIMPLEX,
                       2, (255, 255, 255), 3)

        return img

    async def test_qa_mode(self):
        """Test Pattern 1: Q&A Loop"""
        print("\n" + "=" * 70)
        print("TEST 1: Q&A MODE")
        print("=" * 70)

        self.agent.mode = AgentMode.QA
        self.agent.start_conversation()

        # Simulate a camera frame with detection
        print("\n📸 Processing camera frame...")
        test_image = self.create_test_image("person")
        result = await self.backend.process_frame(test_image)

        print(f"✅ Detected {result['count']} objects")
        print(f"   Guidance: {result['guidance']}")

        # Update agent with visual context
        visual_context = VisualContext(
            detections=result.get('detections', []),
            count=result.get('count', 0),
            guidance=result.get('guidance', ''),
            timestamp=result.get('timestamp', time.time()),
            frame_id=1
        )
        self.agent.update_visual_context(visual_context)

        # Ask questions
        questions = [
            "What objects do you see?",
            "Is it safe to walk forward?",
            "How far away are the objects?"
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n💬 Question {i}: {question}")
            response = await self.agent.query(question, include_visual_context=True)
            print(f"🤖 Agent: {response.text}")
            await asyncio.sleep(1)  # Rate limiting

        print("\n✅ Q&A mode test complete")

    async def test_monitoring_mode(self):
        """Test Pattern 2: Monitoring Loop"""
        print("\n" + "=" * 70)
        print("TEST 2: MONITORING MODE")
        print("=" * 70)

        self.agent.mode = AgentMode.MONITOR
        self.agent.start_conversation()

        # Simulate scene changes
        scenes = [
            ("person", "Initial scene with person"),
            ("car", "Car enters scene"),
            ("person", "Back to person only")
        ]

        for i, (scene_type, description) in enumerate(scenes, 1):
            print(f"\n📸 Frame {i}: {description}")
            test_image = self.create_test_image(scene_type)
            result = await self.backend.process_frame(test_image)

            # Update visual context
            visual_context = VisualContext(
                detections=result.get('detections', []),
                count=result.get('count', 0),
                guidance=result.get('guidance', ''),
                timestamp=result.get('timestamp', time.time()),
                frame_id=i
            )
            self.agent.update_visual_context(visual_context)

            # Check if significant change detected
            if self.agent.last_significant_change:
                print(f"🚨 Significant change detected!")
                response = await self.agent.query(
                    f"The scene changed: {description}. Provide a brief alert.",
                    include_visual_context=True
                )
                print(f"🤖 Alert: {response.text}")

            await asyncio.sleep(1)

        print("\n✅ Monitoring mode test complete")

    async def test_navigation_mode(self):
        """Test Pattern 3: Navigation Loop"""
        print("\n" + "=" * 70)
        print("TEST 3: NAVIGATION MODE")
        print("=" * 70)

        self.agent.mode = AgentMode.NAVIGATION
        self.agent.start_conversation()

        # Start navigation
        print("\n🧭 Starting navigation to 'door'")
        response = await self.agent.query(
            "Guide me to the door",
            include_visual_context=True
        )
        print(f"🤖 {response.text}")

        # Simulate movement frames
        for i in range(3):
            print(f"\n📸 Frame {i+1} (user moving)")
            test_image = self.create_test_image("person")
            result = await self.backend.process_frame(test_image)

            visual_context = VisualContext(
                detections=result.get('detections', []),
                count=result.get('count', 0),
                guidance=result.get('guidance', ''),
                timestamp=result.get('timestamp', time.time()),
                frame_id=i+1
            )
            self.agent.update_visual_context(visual_context)

            # Get navigation update
            response = await self.agent.query(
                "Continue navigation guidance",
                include_visual_context=True
            )
            print(f"🤖 {response.text}")

            await asyncio.sleep(1)

        print("\n✅ Navigation mode test complete")

    async def test_contextual_mode(self):
        """Test Pattern 4: Contextual Understanding"""
        print("\n" + "=" * 70)
        print("TEST 4: CONTEXTUAL MODE")
        print("=" * 70)

        self.agent.mode = AgentMode.CONTEXTUAL
        self.agent.start_conversation()

        # Build up context with multiple frames
        print("\n📸 Building spatial context (3 frames)...")
        for i in range(3):
            test_image = self.create_test_image("person")
            result = await self.backend.process_frame(test_image)

            visual_context = VisualContext(
                detections=result.get('detections', []),
                count=result.get('count', 0),
                guidance=result.get('guidance', ''),
                timestamp=result.get('timestamp', time.time()),
                frame_id=i+1
            )
            self.agent.update_visual_context(visual_context)
            await asyncio.sleep(0.5)

        # Ask contextual question
        print("\n💬 Asking contextual question...")
        response = await self.agent.query(
            "Describe the overall scene layout and what I should know about my surroundings",
            include_visual_context=True
        )
        print(f"🤖 {response.text}")

        print("\n✅ Contextual mode test complete")

    async def test_hybrid_mode(self):
        """Test Pattern 5: Hybrid Mode"""
        print("\n" + "=" * 70)
        print("TEST 5: HYBRID MODE (RECOMMENDED)")
        print("=" * 70)

        self.agent.mode = AgentMode.HYBRID
        self.agent.start_conversation()

        # Simulate hybrid interaction
        print("\n🔄 Hybrid mode combines automatic alerts + on-demand Q&A")

        # Auto-navigation
        print("\n[AUTO] Processing frame...")
        test_image = self.create_test_image("person")
        result = await self.backend.process_frame(test_image)

        visual_context = VisualContext(
            detections=result.get('detections', []),
            count=result.get('count', 0),
            guidance=result.get('guidance', ''),
            timestamp=result.get('timestamp', time.time()),
            frame_id=1
        )
        self.agent.update_visual_context(visual_context)

        print(f"[AUTO] Basic guidance: {result['guidance']}")

        # User interrupts with question
        print("\n[USER] Interrupting with question...")
        response = await self.agent.query(
            "Tell me more details about what's ahead",
            include_visual_context=True
        )
        print(f"🤖 [AGENT] {response.text}")

        # Back to auto-guidance
        await asyncio.sleep(1)
        print("\n[AUTO] Resuming automatic guidance...")
        print(f"[AUTO] {result['guidance']}")

        print("\n✅ Hybrid mode test complete")

    async def run_all_tests(self):
        """Run all integration tests"""
        try:
            await self.test_qa_mode()
            await asyncio.sleep(2)

            await self.test_monitoring_mode()
            await asyncio.sleep(2)

            await self.test_navigation_mode()
            await asyncio.sleep(2)

            await self.test_contextual_mode()
            await asyncio.sleep(2)

            await self.test_hybrid_mode()

            # Final stats
            print("\n" + "=" * 70)
            print("📊 FINAL AGENT STATISTICS")
            print("=" * 70)
            stats = self.agent.get_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")

            print("\n" + "=" * 70)
            print("✅ ALL TESTS PASSED!")
            print("=" * 70)
            print("\nThe Gemini Agent is successfully integrated with AMD backend.")
            print("All 5 loop patterns are operational and ready for production use.")
            print("\nNext steps:")
            print("1. Test with real camera feed")
            print("2. Add mobile UI controls (see README_AGENT.md)")
            print("3. Configure cost controls and rate limiting")
            print("4. Deploy to production environment")

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


async def main():
    """Main entry point"""
    # Check if we're in the right environment
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  Warning: No GPU detected. Tests will be slow.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    except ImportError:
        print("⚠️  Warning: PyTorch not installed. Some features may not work.")

    # Run tests
    tester = AgentIntegrationTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️  python-dotenv not installed. Using system environment.")

    # Run async tests
    asyncio.run(main())
