<!--
  Usage:
  ---
  layout: outro
  ---

  ::takeaways::

  ## Take Aways
  - bullet
  - bullet

  ::right::

  <QRCode text="https://..." class="w-24" />
  **Contact:** ...
  **Code:** ...

  ::footer::

  Funding: ...
-->

<script setup lang="ts">
const props = defineProps({
  class: {
    type: String,
  },
  layoutClass: {
    type: String,
  },
});
</script>

<template>
  <div class="slidev-layout outro relative w-full h-full flex flex-col" :class="props.layoutClass">
    <div class="outro-title text-center mb-4">
      <slot>
        <h1 class="m-0">{{ $slidev?.configs?.title ?? 'Thank You' }}</h1>
      </slot>
    </div>
    <div class="outro-grid flex-1 grid grid-cols-2 gap-6 min-h-0">
      <div class="outro-takeaways overflow-auto" :class="props.class">
        <slot name="takeaways" />
      </div>
      <div class="outro-right flex flex-col items-center justify-center" :class="props.class">
        <slot name="right" />
      </div>
    </div>
    <div v-if="$slots.footer" class="outro-footer absolute bottom-0 left-0 right-0 text-center text-xs opacity-40 py-2">
      <slot name="footer" />
    </div>
  </div>
</template>

<style scoped>
.outro-title :deep(h1) {
  @apply text-3xl;
}
</style>
