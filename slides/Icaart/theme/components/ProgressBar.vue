<template>
    <div class="w-full h-4 bg-gray-300 rounded overflow-hidden">
        <div class="h-full bg-green-500 rounded transition-all duration-1000" :style="{ width: currentProgress + '%' }">
        </div>
    </div>
</template>

<script setup>
import { ref, watch, nextTick, onMounted } from 'vue'
import { onSlideEnter, onSlideLeave } from '@slidev/client'


const props = defineProps({
    progress: {
        type: Number,
        required: true,
        validator: value => value >= 0 && value <= 100,
    },
    annimate: {
        type: Boolean,
        default: false,
    },
})

const currentProgress = ref(0)
if (!props.annimate)
    currentProgress.value = props.progress

onMounted(() => {
    if (!props.annimate) {
        currentProgress.value = props.progress
    }
})

onSlideEnter(async () => {
    // await nextTick()
    currentProgress.value = props.progress
})


// Animate the bar when the progress prop changes
watch(() => props.progress, newVal => {
    currentProgress.value = newVal
})
</script>