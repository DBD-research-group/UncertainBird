<template>
    <img :src="caulogo" alt="Kiel University" class="absolute left-[0.5vw] top-[0.5vw] h-[1.5em]" />
    <img :src="inslogo" alt="Intelligent Systems" class="absolute right-[0.5vw] top-[0.5vw] h-[1.5em]" />
</template>

<script setup lang="ts">
import inslogo from '../public/ins-logo.png'
import caulogo from '../public/cau-lila.png'
</script>