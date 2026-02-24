<template>
  <figure class="flex flex-col items-center justify-center">
    <PdfFigure
      v-if="isPdf"
      :src="resolveAssetUrl(url)"
      class="max-h-full max-w-full"
    />
    <img
      v-else
      :alt="caption"
      class="max-h-full"
      :src="resolveAssetUrl(url)"
    />
    <figcaption class="mt-3 text-center text-xs" v-if="caption">
      <span v-html="renderedCaption" /><sup v-if="footnoteNumber">{{ footnoteNumber }}</sup>
    </figcaption>
  </figure>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import katex from 'katex';
import { resolveAssetUrl } from '../layout-helper';

const props = defineProps<{
  caption?: string;
  footnoteNumber?: number;
  url: string;
}>();

const isPdf = computed(() => props.url.toLowerCase().endsWith('.pdf'));

const renderedCaption = computed(() => {
  if (!props.caption) return '';
  return props.caption.replace(/\$([^$]+)\$/g, (_, math) =>
    katex.renderToString(math, { throwOnError: false }),
  );
});
</script>
