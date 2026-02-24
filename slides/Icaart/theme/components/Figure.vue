<template>
    <img :src="props.src" :alt="props.alt" class="rounded-lg mx-auto" :class="props.class" />
    <p class="mx-auto text-center"> <span class="color-main">Figure</span>:
        <slot />
    </p>
</template>ed

<script setup lang="ts">
const props = defineProps<{
    src: string
    alt: string
    class: string
}>()

</script>