<template>
  <Pagination v-if="showPagination"
    :x="themeConfigs?.paginationX"
    :y="themeConfigs?.paginationY"
    :author="configs?.author"
  />
  <div
    v-if="showPresenter"
    class="absolute bottom-0 left-1/2 -translate-x-1/2 px-3 py-1 text-xs text-gray-800/40 pointer-events-none select-none w-full text-center"
  >
    {{ configs?.presenter }}
  </div>
  <Logos v-if="showLogos" />
  <CiteFootnotes />
</template>

<script setup lang="ts">
import { useSlideContext } from '@slidev/client';
import { useCitations } from './composables/useCitations';
import { computed } from 'vue';

const { $slidev, $frontmatter, $route } = useSlideContext();
const citations = useCitations();

const configs = computed(() => $slidev.configs);
const themeConfigs = computed(() => $slidev.themeConfigs);
const currentPage = computed(() => $slidev.nav.currentPage);
const slideFootnotes = computed(() => citations.getSlideFootnotes($route.no));

const isDisabledPage = computed(() =>
  themeConfigs.value?.paginationPagesDisabled?.includes(currentPage.value)
);

const showPagination = computed(() =>
  currentPage.value !== $slidev.nav.total + 1 &&
  !isDisabledPage.value &&
  !!(themeConfigs.value?.paginationX || themeConfigs.value?.paginationY) &&
  ($frontmatter.pagination === undefined || $frontmatter.pagination === true)
);

const showPresenter = computed(() =>
  !!configs.value?.presenter &&
  !isDisabledPage.value &&
  $frontmatter.layout !== 'outro' &&
  slideFootnotes.value.length === 0
);

const showLogos = computed(() =>
  $frontmatter.logos === undefined || $frontmatter.logos === true
);
</script>
