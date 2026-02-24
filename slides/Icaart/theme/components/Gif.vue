<template>
    <img :src="src" :alt="props.alt" class="rounded-lg mx-auto" :class="props.class" />
    <p v-if="caption" class="mx-auto text-center"> <span class="color-main">Figure</span>:
        <slot />
    </p>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { onSlideEnter, useIsSlideActive } from '@slidev/client'


const props = defineProps<{
    src: string;
    alt: string;
    class: string;
    caption: Boolean;
}>();

const src = ref(props.src)


onSlideEnter(() => {
    // reset the gif
    src.value = props.src + '?' + Math.random()
})


</script>
