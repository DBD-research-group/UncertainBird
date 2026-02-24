<template>
  <sup class="cite-ref">[{{ number }}]</sup>
</template>

<script setup lang="ts">
import { useSlideContext } from '@slidev/client';
import { useCitations } from '../composables/useCitations';
import { onMounted } from 'vue';

const props = defineProps<{
  refKey: string;
}>();

const { $slidev, $route } = useSlideContext();
const citations = useCitations();
const slideNo = $route.no;
const number = citations.registerCitation(slideNo, props.refKey);

onMounted(() => {
  const bibFile = ($slidev.configs?.bibFile as string) || 'references.bib';
  citations.loadBib(bibFile);
});
</script>

<style scoped>
.cite-ref {
  font-size: 0.75em;
  opacity: 0.85;
}
</style>
