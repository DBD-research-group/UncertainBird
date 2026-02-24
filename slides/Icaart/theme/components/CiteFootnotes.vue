<template>
  <div
    v-if="footnotes.length > 0"
    class="cite-footnotes absolute bottom-0 left-0 w-full px-3 py-2 text-xs opacity-60"
  >
    <div
      v-for="fn in footnotes"
      :key="fn.num"
      class="cite-footnote-item"
    >
      <a
        v-if="fn.url"
        :href="fn.url"
        target="_blank"
        rel="noopener noreferrer"
        class="cite-footnote-link"
      >[{{ fn.num }}] {{ fn.text }}</a>
      <template v-else>[{{ fn.num }}] {{ fn.text }}</template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useSlideContext } from '@slidev/client';
import { useCitations } from '../composables/useCitations';
import { computed, onMounted } from 'vue';

const { $slidev, $route } = useSlideContext();
const citations = useCitations();
const slideNo = $route.no;

const footnotes = computed(() => citations.getSlideFootnotes(slideNo));

onMounted(() => {
  const bibFile = ($slidev.configs?.bibFile as string) || 'references.bib';
  citations.loadBib(bibFile);
});
</script>

<style scoped>
.cite-footnotes {
  z-index: 1;
  line-height: 1.3;
}
.cite-footnote-item {
  white-space: normal;
}
.cite-footnote-link {
  color: inherit;
  text-decoration: underline;
}
.cite-footnote-link:hover {
  opacity: 1;
}
</style>
