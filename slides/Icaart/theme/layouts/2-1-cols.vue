<!--
  Usage:

```md
---
layout: 2-1-cols
---

# Left

This shows on the left

::right::

# Right

This shows on the right
```

-->

<script setup lang="ts">
const props = defineProps({
    class: {
        type: String,
    },
    layoutClass: {
        type: String,
    },
})
</script>

<template>
    <div class="slidev-layout two-columns w-full h-full grid grid-cols-3" :class="props.layoutClass">
        <div class="col-left col-span-2" :class="props.class">
            <slot />
        </div>
        <div class="col-right" :class="props.class">
            <slot name="right" />
        </div>
    </div>
</template>