<template>
  <canvas ref="canvasRef" />
</template>

<script setup lang="ts">
import { ref, onMounted, watch, onBeforeUnmount } from 'vue';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
import type { PDFDocumentProxy } from 'pdfjs-dist';

GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).href;

const props = withDefaults(
  defineProps<{
    src: string;
    page?: number;
    /** Render resolution multiplier (higher = crisper). */
    scale?: number;
  }>(),
  { page: 1, scale: 3 },
);

const canvasRef = ref<HTMLCanvasElement>();
let pdfDoc: PDFDocumentProxy | null = null;

async function render() {
  const canvas = canvasRef.value;
  if (!canvas) return;

  pdfDoc?.destroy();
  pdfDoc = await getDocument(props.src).promise;
  const page = await pdfDoc.getPage(props.page);
  const viewport = page.getViewport({ scale: props.scale });

  canvas.width = viewport.width;
  canvas.height = viewport.height;

  const ctx = canvas.getContext('2d')!;
  await page.render({ canvasContext: ctx, viewport }).promise;
}

onMounted(render);
watch(() => props.src, render);
onBeforeUnmount(() => pdfDoc?.destroy());
</script>
